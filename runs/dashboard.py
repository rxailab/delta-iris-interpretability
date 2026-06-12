#!/usr/bin/env python3
"""Live dashboard for a Δ-IRIS (or IRIS) Slurm training job.

Tails the job's .out file and shows: job state, current epoch, current stage,
recent throughput, and ETA. Uses only stdlib + ANSI escapes.

    python dashboard.py                 # auto-pick latest delta-iris-full-* job
    python dashboard.py 21531177        # specific Slurm job id
    python dashboard.py path/to/job.out # any .out file
    python dashboard.py --no-loop       # one-shot snapshot
    python dashboard.py --interval 10   # refresh every 10s (default 5)
    python dashboard.py --serve         # serve HTML on http://127.0.0.1:8765
    python dashboard.py --serve --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS_DIR = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/runs")

# ANSI codes
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GRN, YEL, BLU, MAG, CYN = (f"\033[{i}m" for i in (31, 32, 33, 34, 35, 36))
CLEAR = "\033[H\033[J"

STAGE_RE = re.compile(
    r"(Experience collection \([^)]+\)|Training (?:tokenizer|world_model|actor_critic)):\s+"
    r"(\d+)%\|[^|]*\|\s*([\d.]+)(?:k)?(?:it)?/?(\d+|\?)(?:it)?\s*"
    r"\[([\d:]+)<([\d:?]+),\s*([\d.]+)\s*it/s\]"
)
# Simpler: capture any tqdm line of the form
# "<desc>:   N%|...|  done/total [elapsed<remaining, X it/s]"
TQDM_RE = re.compile(
    r"^(?P<desc>[A-Za-z][^:]+):\s+"
    r"(?P<pct>\d+)%\|[^|]*\|\s*"
    r"(?P<done>\d+(?:it)?)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:?]+),\s*(?P<rate>[\d.]+)\s*it/s\]"
)
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*/\s*(\d+)\s*$")
STAGE_NAMES = {
    "Experience collection (train_dataset)": "collect",
    "Training tokenizer": "tokenizer",
    "Training world_model": "world_model",
    "Training actor_critic": "actor_critic",
}


def hms(seconds: float | int) -> str:
    seconds = int(max(0, seconds))
    d, seconds = divmod(seconds, 86400)
    h, seconds = divmod(seconds, 3600)
    m, s = divmod(seconds, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def parse_clock(s: str) -> int:
    # tqdm formats: "MM:SS" or "H:MM:SS" or "?"
    if s == "?" or ":" not in s:
        return 0
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + sec


def latest_full_job() -> tuple[int | None, Path | None]:
    """Pick the most recently modified iris-* / delta-iris-* .out file."""
    candidates = set()
    for pattern in ("delta-iris-*.out", "iris-*.out"):
        candidates.update(RUNS_DIR.glob(pattern))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        m = re.search(r"-(\d+)\.out$", p.name)
        if m:
            return int(m.group(1)), p
    return None, None


def slurm_status(job_id: int) -> dict[str, str]:
    """Return State / Elapsed / NodeList / Partition / TimeLimit for a job, via
    squeue first (cheap, current) and sacct as fallback."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", str(job_id),
             "-o", "%T|%M|%R|%P|%l"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            state, elapsed, node, partition, time_limit = out.split("|")
            return dict(state=state, elapsed=elapsed, node=node,
                        partition=partition, time_limit=time_limit)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["sacct", "-Pn", "-j", str(job_id),
             "-o", "State,Elapsed,NodeList,Partition,Timelimit"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            line = out.split("\n")[0]
            state, elapsed, node, partition, time_limit = line.split("|")
            return dict(state=state, elapsed=elapsed, node=node,
                        partition=partition, time_limit=time_limit)
    except Exception:
        pass
    return dict(state="?", elapsed="?", node="?", partition="?", time_limit="?")


def tail_lines(path: Path, max_bytes: int = 4_000_000) -> list[str]:
    """Read last max_bytes of file, split on \\r and \\n so we don't drop tqdm
    in-place updates."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # skip partial line
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    # tqdm uses \r to update in place. Split on either \r or \n.
    return [ln for ln in re.split(r"[\r\n]", text) if ln]


def parse_log(path: Path) -> dict:
    """Extract structured status from the .out file."""
    lines = tail_lines(path)
    state = {
        "epoch_curr": None,
        "epoch_total": None,
        "stage": None,
        "stage_done": None,
        "stage_total": None,
        "stage_pct": None,
        "stage_rate": None,
        "stage_elapsed_s": 0,
        "stage_remaining_s": 0,
        "completed_stages": [],  # list of (epoch, stage, elapsed_s)
        "last_line": "",
        "exit_line": None,
    }

    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        state["last_line"] = line

        m = EPOCH_RE.match(line)
        if m:
            state["epoch_curr"] = int(m.group(1))
            state["epoch_total"] = int(m.group(2))
            continue

        if line.startswith("=== full run exit") or line.startswith("=== bench exit") or line.startswith("=== smoke train exit"):
            state["exit_line"] = line
            continue

        m = TQDM_RE.match(line)
        if not m:
            continue
        desc = m.group("desc")
        stage = STAGE_NAMES.get(desc) or desc
        done = int(re.sub(r"\D", "", m.group("done")))
        total = int(m.group("total"))
        pct = int(m.group("pct"))
        rate = float(m.group("rate"))
        elapsed = parse_clock(m.group("elapsed"))
        remaining = parse_clock(m.group("remaining"))

        state["stage"] = stage
        state["stage_done"] = done
        state["stage_total"] = total
        state["stage_pct"] = pct
        state["stage_rate"] = rate
        state["stage_elapsed_s"] = elapsed
        state["stage_remaining_s"] = remaining

        if pct == 100 and state["epoch_curr"] is not None:
            entry = (state["epoch_curr"], stage, elapsed)
            if not state["completed_stages"] or state["completed_stages"][-1] != entry:
                state["completed_stages"].append(entry)

    return state


def epoch_throughput(state: dict) -> tuple[float | None, int]:
    """Average seconds per epoch over the last few completed epochs.

    A "completed epoch" = saw all four stages (collect, tokenizer, world_model,
    actor_critic). Sum their elapsed times.
    """
    by_epoch: dict[int, dict[str, int]] = {}
    for epoch, stage, elapsed in state["completed_stages"]:
        by_epoch.setdefault(epoch, {})[stage] = elapsed
    durations = []
    for ep in sorted(by_epoch):
        stages = by_epoch[ep]
        # Need at least tokenizer+wm+ac (collection might be skipped past epoch 991)
        if {"tokenizer", "world_model", "actor_critic"}.issubset(stages):
            durations.append(sum(stages.values()))
    if not durations:
        return None, 0
    last = durations[-10:]
    return sum(last) / len(last), len(durations)


def progress_bar(pct: float, width: int = 30) -> str:
    pct = max(0.0, min(100.0, pct))
    full = int(width * pct / 100)
    return "[" + "█" * full + "░" * (width - full) + "]"


def render(state: dict, slurm: dict, log_path: Path, job_id: int | None) -> str:
    cols = shutil.get_terminal_size((100, 30)).columns
    lines: list[str] = []
    title = f" Δ-IRIS Training Dashboard "
    bar = "═" * ((cols - len(title)) // 2)
    lines.append(f"{BOLD}{CYN}{bar}{title}{bar}{RESET}")
    job_str = f"job {job_id}" if job_id else "log file"
    lines.append(
        f"{job_str}  state={BOLD}{slurm['state']}{RESET}  "
        f"node={slurm['node']}  partition={slurm['partition']}  "
        f"elapsed={slurm['elapsed']} / limit={slurm['time_limit']}"
    )
    lines.append(f"{DIM}{log_path}{RESET}")
    lines.append("")

    epc, ept = state["epoch_curr"], state["epoch_total"]
    if epc and ept:
        ep_pct = 100.0 * (epc - 1) / ept  # epoch N is in progress, so done = N-1
        lines.append(
            f"{BOLD}Epoch{RESET}    {epc:4d} / {ept}    "
            f"{progress_bar(ep_pct)}  {ep_pct:5.1f}%"
        )
    else:
        lines.append(f"{BOLD}Epoch{RESET}    (not started yet)")

    stage = state["stage"]
    if stage:
        rate = state["stage_rate"] or 0
        spd = f"{rate:.2f} it/s" if rate < 100 else f"{rate:.0f} it/s"
        col = {"collect": YEL, "tokenizer": GRN,
               "world_model": BLU, "actor_critic": MAG}.get(stage, "")
        lines.append(
            f"{BOLD}Stage{RESET}    {col}{stage:<13s}{RESET} "
            f"{progress_bar(state['stage_pct'] or 0, width=24)} "
            f"{state['stage_done']}/{state['stage_total']}  "
            f"{spd}  ({hms(state['stage_elapsed_s'])} so far, "
            f"~{hms(state['stage_remaining_s'])} left)"
        )
    else:
        lines.append(f"{BOLD}Stage{RESET}    (idle)")

    avg, n = epoch_throughput(state)
    if avg and epc and ept:
        remaining_epochs = ept - epc + 1  # current one not finished
        eta_s = remaining_epochs * avg
        finish_at = datetime.now() + timedelta(seconds=eta_s)
        lines.append("")
        lines.append(
            f"{BOLD}Throughput{RESET}  {avg/60:.2f} min/epoch  "
            f"({DIM}avg of last {min(n,10)} completed epochs{RESET})"
        )
        lines.append(
            f"{BOLD}ETA{RESET}         {hms(eta_s)} remaining  "
            f"→ finish ~ {finish_at.strftime('%Y-%m-%d %H:%M')}"
        )

    if state["completed_stages"]:
        lines.append("")
        lines.append(f"{BOLD}Recent completed stages:{RESET}")
        seen = set()
        for epoch, stg, el in reversed(state["completed_stages"]):
            key = (epoch, stg)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  epoch {epoch:4d}  {stg:<13s}  {hms(el)}")
            if len(seen) >= 6:
                break

    if state["exit_line"]:
        lines.append("")
        lines.append(f"{BOLD}{GRN}{state['exit_line']}{RESET}")

    lines.append("")
    lines.append(f"{DIM}refreshed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                 f" — Ctrl-C to quit{RESET}")
    return "\n".join(lines)


def resolve_log(arg: str | None) -> tuple[int | None, Path]:
    if arg is None:
        jid, path = latest_full_job()
        if path is None:
            sys.exit("no delta-iris-* / iris-* .out files found in runs/")
        return jid, path
    if arg.isdigit():
        jid = int(arg)
        path = RUNS_DIR / f"delta-iris-full-{jid}.out"
        if not path.exists():
            matches = list(RUNS_DIR.glob(f"*-{jid}.out"))
            if not matches:
                sys.exit(f"no .out file found for job {jid}")
            path = matches[0]
        return jid, path
    p = Path(arg)
    if not p.exists():
        sys.exit(f"file not found: {p}")
    m = re.search(r"-(\d+)\.out$", p.name)
    return (int(m.group(1)) if m else None), p


def collect_state(job_id: int | None, log_path: Path) -> dict:
    """Pull everything the HTML/JSON view needs into a single dict."""
    state = parse_log(log_path)
    slurm = slurm_status(job_id) if job_id else dict(
        state="?", elapsed="?", node="?", partition="?", time_limit="?",
    )
    avg_s, n_epochs = epoch_throughput(state)
    epc, ept = state["epoch_curr"], state["epoch_total"]
    eta_s = None
    finish_at = None
    if avg_s and epc and ept:
        eta_s = (ept - epc + 1) * avg_s
        finish_at = (datetime.now() + timedelta(seconds=eta_s)).isoformat(timespec="seconds")

    # Deduplicate recent completed stages newest-first.
    seen: set = set()
    recent: list[dict] = []
    for ep, stg, el in reversed(state["completed_stages"]):
        key = (ep, stg)
        if key in seen:
            continue
        seen.add(key)
        recent.append(dict(epoch=ep, stage=stg, elapsed_s=el, elapsed_hms=hms(el)))
        if len(recent) >= 8:
            break

    return dict(
        job_id=job_id,
        log_path=str(log_path),
        now=datetime.now().isoformat(timespec="seconds"),
        slurm=slurm,
        epoch_curr=epc,
        epoch_total=ept,
        epoch_pct=(100.0 * (epc - 1) / ept) if epc and ept else None,
        stage=state["stage"],
        stage_done=state["stage_done"],
        stage_total=state["stage_total"],
        stage_pct=state["stage_pct"],
        stage_rate=state["stage_rate"],
        stage_elapsed_s=state["stage_elapsed_s"],
        stage_elapsed_hms=hms(state["stage_elapsed_s"] or 0),
        stage_remaining_s=state["stage_remaining_s"],
        stage_remaining_hms=hms(state["stage_remaining_s"] or 0),
        throughput_s_per_epoch=avg_s,
        throughput_min_per_epoch=(avg_s / 60.0) if avg_s else None,
        throughput_n_epochs=n_epochs,
        eta_s=eta_s,
        eta_hms=hms(eta_s) if eta_s else None,
        finish_at=finish_at,
        recent_stages=recent,
        exit_line=state["exit_line"],
    )


HTML_PAGE = """<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<title>Δ-IRIS dashboard</title>
<style>
:root {
  --bg: #0d1117; --panel: #161b22; --border: #30363d;
  --text: #c9d1d9; --muted: #8b949e;
  --collect: #d29922; --tokenizer: #3fb950;
  --world_model: #58a6ff; --actor_critic: #d2a8ff;
  --ok: #3fb950; --warn: #d29922; --err: #f85149;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
       font-size: 14px; padding: 20px; }
h1 { margin: 0 0 4px; font-size: 18px; color: #79c0ff; }
.muted { color: var(--muted); }
.row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.card { background: var(--panel); border: 1px solid var(--border);
        border-radius: 6px; padding: 14px 16px; flex: 1 1 280px; min-width: 280px; }
.card h2 { margin: 0 0 8px; font-size: 12px; letter-spacing: .12em;
           text-transform: uppercase; color: var(--muted); font-weight: 600; }
.big { font-size: 24px; font-weight: 600; }
.bar { background: #21262d; border-radius: 4px; height: 8px; overflow: hidden;
       margin-top: 8px; }
.bar > div { height: 100%; background: #58a6ff; transition: width .4s ease; }
.bar.collect > div { background: var(--collect); }
.bar.tokenizer > div { background: var(--tokenizer); }
.bar.world_model > div { background: var(--world_model); }
.bar.actor_critic > div { background: var(--actor_critic); }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px;
      align-items: baseline; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; }
.badge { padding: 2px 8px; border-radius: 999px; font-size: 11px;
         font-weight: 600; display: inline-block; }
.badge.RUNNING { background: #163b1b; color: var(--ok); }
.badge.PENDING { background: #3a2a06; color: var(--warn); }
.badge.COMPLETED { background: #0d2818; color: var(--ok); }
.badge.FAILED, .badge.CANCELLED, .badge.TIMEOUT
  { background: #3b1018; color: var(--err); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
table th, table td { padding: 6px 10px; text-align: left;
                     border-bottom: 1px solid var(--border); }
table th { color: var(--muted); font-weight: 600;
           text-transform: uppercase; letter-spacing: .08em; font-size: 11px; }
table tr:last-child td { border-bottom: none; }
.stage-collect { color: var(--collect); }
.stage-tokenizer { color: var(--tokenizer); }
.stage-world_model { color: var(--world_model); }
.stage-actor_critic { color: var(--actor_critic); }
.footer { color: var(--muted); margin-top: 16px; font-size: 12px; }
</style>
</head>
<body>
<h1>Δ-IRIS Training Dashboard</h1>
<div class=muted id=topline>loading…</div>

<div class=row>
  <div class=card>
    <h2>Job</h2>
    <dl class=kv>
      <dt>state</dt><dd id=jstate>—</dd>
      <dt>job id</dt><dd id=jid>—</dd>
      <dt>node</dt><dd id=jnode>—</dd>
      <dt>partition</dt><dd id=jpart>—</dd>
      <dt>elapsed</dt><dd id=jelapsed>—</dd>
      <dt>time limit</dt><dd id=jlimit>—</dd>
    </dl>
  </div>
  <div class=card>
    <h2>Epoch</h2>
    <div class=big><span id=epoch>—</span> / <span id=etotal>—</span></div>
    <div class=bar><div id=epoch_bar style="width:0%"></div></div>
    <div class=muted style="margin-top:6px"><span id=epoch_pct>0</span>%</div>
  </div>
  <div class=card>
    <h2>Stage</h2>
    <div class=big><span id=stage_name>—</span></div>
    <div class=bar id=stage_bar_outer><div id=stage_bar style="width:0%"></div></div>
    <div class=muted style="margin-top:6px">
      <span id=stage_done>—</span>/<span id=stage_total>—</span>
      &nbsp;·&nbsp; <span id=stage_rate>—</span> it/s
      &nbsp;·&nbsp; <span id=stage_elapsed>—</span> elapsed,
      <span id=stage_remaining>—</span> left
    </div>
  </div>
</div>

<div class=row>
  <div class=card>
    <h2>Throughput</h2>
    <div class=big><span id=tp>—</span> <span class=muted style="font-size:14px">min/epoch</span></div>
    <div class=muted style="margin-top:6px">avg of last <span id=tp_n>0</span> epochs</div>
  </div>
  <div class=card>
    <h2>ETA</h2>
    <div class=big id=eta>—</div>
    <div class=muted style="margin-top:6px">finish ~ <span id=finish>—</span></div>
  </div>
  <div class=card>
    <h2>Recent stages</h2>
    <table id=recent_tbl><thead><tr><th>epoch</th><th>stage</th><th>elapsed</th></tr></thead><tbody></tbody></table>
  </div>
</div>

<div class=footer>
  <span id=now>—</span>
  &nbsp;·&nbsp; refreshes every <span id=interval>5</span>s
  &nbsp;·&nbsp; <span class=muted id=logpath></span>
</div>

<script>
const INTERVAL_MS = 5000;
const $ = (id) => document.getElementById(id);

function set(id, v) { $(id).textContent = (v === null || v === undefined) ? "—" : v; }

function fmtNum(v, fmt) {
  if (v === null || v === undefined) return "—";
  if (fmt === "pct1") return v.toFixed(1);
  if (fmt === "min2") return v.toFixed(2);
  if (fmt === "int") return Math.floor(v).toString();
  return v.toString();
}

async function refresh() {
  let r;
  try {
    r = await fetch("/api/state", {cache: "no-store"});
  } catch (e) {
    set("topline", "fetch error: " + e);
    return;
  }
  if (!r.ok) { set("topline", "HTTP " + r.status); return; }
  const s = await r.json();

  // Top line
  set("topline", `${s.log_path}`);
  set("logpath", s.log_path);
  set("now", "refreshed " + s.now);
  $("interval").textContent = (INTERVAL_MS / 1000).toString();

  // Job card
  const st = (s.slurm && s.slurm.state) || "?";
  $("jstate").innerHTML = `<span class="badge ${st}">${st}</span>`;
  set("jid", s.job_id || "—");
  set("jnode", s.slurm.node);
  set("jpart", s.slurm.partition);
  set("jelapsed", s.slurm.elapsed);
  set("jlimit", s.slurm.time_limit);

  // Epoch card
  set("epoch", s.epoch_curr ?? "—");
  set("etotal", s.epoch_total ?? "—");
  const pct = s.epoch_pct ?? 0;
  $("epoch_bar").style.width = pct + "%";
  set("epoch_pct", fmtNum(pct, "pct1"));

  // Stage card
  set("stage_name", s.stage ?? "—");
  $("stage_bar_outer").className = "bar " + (s.stage || "");
  $("stage_bar").style.width = (s.stage_pct ?? 0) + "%";
  set("stage_done", s.stage_done ?? "—");
  set("stage_total", s.stage_total ?? "—");
  set("stage_rate", s.stage_rate ? s.stage_rate.toFixed(2) : "—");
  set("stage_elapsed", s.stage_elapsed_hms);
  set("stage_remaining", s.stage_remaining_hms);

  // Throughput / ETA
  set("tp", s.throughput_min_per_epoch ? s.throughput_min_per_epoch.toFixed(2) : "—");
  set("tp_n", s.throughput_n_epochs);
  set("eta", s.eta_hms ?? "—");
  set("finish", s.finish_at ?? "—");

  // Recent stages
  const tbody = $("recent_tbl").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of (s.recent_stages || [])) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.epoch}</td>
                    <td class="stage-${r.stage}">${r.stage}</td>
                    <td>${r.elapsed_hms}</td>`;
    tbody.appendChild(tr);
  }
}

refresh();
setInterval(refresh, INTERVAL_MS);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    # Set by the server before serving.
    job_id: int | None = None
    log_path: Path | None = None
    auto: bool = False  # if True, re-resolve the latest job on every request

    def log_message(self, format, *args):  # silence access log
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/state":
            try:
                job_id, log_path = self.job_id, self.log_path
                if self.auto:
                    # Follow the newest run (handles resumes spawning new jobs).
                    jid, lp = latest_full_job()
                    if lp is not None:
                        job_id, log_path = jid, lp
                state = collect_state(job_id, log_path)
                body = json.dumps(state).encode("utf-8")
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
            else:
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def serve_mode(host: str, port: int, job_id: int | None, log_path: Path,
               auto: bool = False) -> None:
    DashboardHandler.job_id = job_id
    DashboardHandler.log_path = log_path
    DashboardHandler.auto = auto
    httpd = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Serving Δ-IRIS dashboard at http://{host}:{port}/ "
          f"(job={job_id}, log={log_path}, auto-follow={auto})", flush=True)
    print(f"  SSH tunnel from your laptop:", flush=True)
    print(f"    ssh -N -L {port}:localhost:{port} <your-hpc-host>", flush=True)
    print(f"  then open http://localhost:{port}/ in your browser.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?",
                        help="job id, .out path, or omit for latest")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="refresh interval in seconds (default 5)")
    parser.add_argument("--no-loop", action="store_true",
                        help="print one snapshot and exit")
    parser.add_argument("--serve", action="store_true",
                        help="serve HTML dashboard over HTTP")
    parser.add_argument("--port", type=int, default=8765,
                        help="port for --serve (default 8765)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind host for --serve (default 127.0.0.1)")
    args = parser.parse_args()

    job_id, log_path = resolve_log(args.target)

    if args.serve:
        # When no explicit target is given, keep following the newest run so the
        # dashboard survives resume jobs that get a fresh Slurm id.
        serve_mode(args.host, args.port, job_id, log_path, auto=(args.target is None))
        return

    try:
        while True:
            state = parse_log(log_path)
            slurm = slurm_status(job_id) if job_id else dict(
                state="?", elapsed="?", node="?", partition="?", time_limit="?",
            )
            screen = render(state, slurm, log_path, job_id)
            if args.no_loop:
                print(screen)
                return
            print(CLEAR + screen, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
