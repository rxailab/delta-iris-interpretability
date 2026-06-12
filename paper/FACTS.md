# FACT SHEET — Δ-IRIS Interpretability (verified 2026-06-12)

Every number below was recomputed or copied verbatim from the named source file
on 2026-06-12 using `/mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env/bin/python`.
Values are given to >=3 significant figures; longer expansions are kept where the
source file provides them. Where a number could not be computed it is flagged
explicitly.

Path abbreviations:
- `results/` = `/mmfs1/storage/users/xiar3/exp/ExpWM/results/`
- `scratch/` = `/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393/`
- `runs/` = `/mmfs1/storage/users/xiar3/exp/ExpWM/runs/`

---

## 1. AGENT / TRAINING

Source: recomputed from `scratch/rollouts.npz` (arrays: `episode_returns`,
`episode_lengths`, `ach_just_unlocked`, `episode_ids`, `achievement_names`, `tokens`).

- n episodes: **500**
- total environment steps: **294,916** (`tokens.shape[0]`; equals `episode_lengths.sum()`)
- mean episode length: 294,916 / 500 = **589.832**
- episode return: **mean = 16.2000** (16.200002670288086),
  **std = 1.6143** (population, ddof=0) / **1.6159** (sample, ddof=1);
  SEM (ddof=1) = 0.0723; min = 5.100, max = 18.100
- mean achievements unlocked per episode (sum of table below / 500): **17.10**

Per-achievement unlock counts out of 500 episodes (an episode counts as
unlocking achievement *a* if any of its steps has the `ach_just_unlocked[:,a]`
bit set; episodes grouped via `episode_ids`). All 22 achievements:

| achievement | episodes unlocked /500 |
|---|---|
| collect_coal | 454 |
| collect_diamond | 0 |
| collect_drink | 483 |
| collect_iron | 366 |
| collect_sapling | 499 |
| collect_stone | 492 |
| collect_wood | 500 |
| defeat_skeleton | 438 |
| defeat_zombie | 487 |
| eat_cow | 489 |
| eat_plant | 0 |
| make_iron_pickaxe | 2 |
| make_iron_sword | 2 |
| make_stone_pickaxe | 478 |
| make_stone_sword | 441 |
| make_wood_pickaxe | 493 |
| make_wood_sword | 485 |
| place_furnace | 482 |
| place_plant | 499 |
| place_stone | 489 |
| place_table | 500 |
| wake_up | 471 |

Never unlocked: collect_diamond, eat_plant (0/500 each). No per-achievement
statistics exist for these two anywhere in the pipeline (and only n=2 for
make_iron_pickaxe / make_iron_sword).

---

## 2. CODEBOOK

Sources: `results/meta.json` (counts of scan) and recomputed from
`scratch/codebook_stats.npz` (array `counts`, shape (4 slots, 1024 codes)).

From `meta.json`:
- transitions scanned: **9,980,073**
- episodes scanned: **29,136**
- K_slots = 4, codebook size C = 1024, embed_dim = 64
- `n_active_codes_total` = 1024; `n_active_codes_per_slot` = [975, 972, 931, 911]

Recomputed from `codebook_stats.npz` (`counts > 0`), confirming meta.json:
- active codes per slot: **slot 0: 975, slot 1: 972, slot 2: 931, slot 3: 911**
- union of active codes across slots: **1024 / 1024** (every code used somewhere)
- each slot's counts sum to 9,980,073 (= transitions scanned)

Per-slot distribution stats (p = counts/total per slot; H = -sum p log2 p over
nonzero counts):

| slot | entropy (bits) | max single-code share | argmax code | top-10 share |
|---|---|---|---|---|
| 0 | 5.9679 | 0.31912 | 29 | 0.54843 |
| 1 | 5.9610 | 0.32353 | 29 | 0.54624 |
| 2 | 5.3359 | 0.47075 | 29 | 0.56012 |
| 3 | 5.6572 | 0.41829 | 29 | 0.54376 |

Note: the dominant code in every slot is code 29 (the "nothing changed in this
quadrant" code); slot 2's takes 47.07% of all mass.

---

## 3. MI TABLE

Source: `results/mi/mi_table.json` (list of records with fields achievement,
slot, code, n_unlock, n_co, n_code, p_a_given_c, lift, pvalue).

- number of records: **431**
- achievements represented: 18 (the four with n<=2 unlocks are absent)

Top-8 rows by lift:

| achievement | slot | code | n_co | n_code | n_unlock | P(a|c) | lift | p-value |
|---|---|---|---|---|---|---|---|---|
| collect_iron | 3 | 75 | 360 | 410 | 366 | 0.8780 | 707.52 | 0.0 |
| place_plant | 3 | 938 | 477 | 519 | 499 | 0.9191 | 543.19 | 0.0 |
| collect_sapling | 2 | 707 | 419 | 463 | 499 | 0.9050 | 534.85 | 0.0 |
| collect_sapling | 3 | 379 | 494 | 559 | 499 | 0.8837 | 522.29 | 0.0 |
| collect_coal | 3 | 578 | 450 | 626 | 454 | 0.7188 | 466.96 | 0.0 |
| make_wood_pickaxe | 3 | 903 | 19 | 35 | 493 | 0.5429 | 324.74 | 6.88e-44 |
| make_wood_sword | 3 | 903 | 14 | 35 | 485 | 0.4000 | 243.23 | 2.38e-30 |
| wake_up | 1 | 583 | 149 | 400 | 471 | 0.3725 | 233.24 | 2.74e-304 |

All rows involving (slot 3, code 36) — the polysemantic "wood-craft" code
(3 rows; n_code = 1507 in all):

| achievement | n_co | n_code | n_unlock | P(a|c) | lift | p-value |
|---|---|---|---|---|---|---|
| place_table | 480 | 1507 | 500 | 0.3185 | 187.87 | 0.0 |
| make_wood_sword | 385 | 1507 | 485 | 0.2555 | 155.35 | 0.0 |
| make_wood_pickaxe | 383 | 1507 | 493 | 0.2541 | 152.03 | 0.0 |

Best row per achievement by P(a|c) (all 18 achievements present in table):

| achievement | slot | code | n_co | n_code | P(a|c) | lift | p-value |
|---|---|---|---|---|---|---|---|
| collect_coal | 3 | 578 | 450 | 626 | 0.7188 | 466.96 | 0.0 |
| collect_drink | 2 | 991 | 66 | 766 | 0.0862 | 52.61 | 1.05e-88 |
| collect_iron | 3 | 75 | 360 | 410 | 0.8780 | 707.52 | 0.0 |
| collect_sapling | 2 | 707 | 419 | 463 | 0.9050 | 534.85 | 0.0 |
| collect_stone | 3 | 991 | 482 | 4315 | 0.1117 | 66.96 | 0.0 |
| collect_wood | 3 | 940 | 474 | 2471 | 0.1918 | 113.14 | 0.0 |
| defeat_skeleton | 0 | 941 | 20 | 301 | 0.0664 | 44.74 | 1.47e-26 |
| defeat_zombie | 1 | 727 | 68 | 365 | 0.1863 | 112.82 | 3.54e-115 |
| eat_cow | 2 | 648 | 251 | 1127 | 0.2227 | 134.32 | 0.0 |
| make_stone_pickaxe | 3 | 616 | 96 | 613 | 0.1566 | 96.62 | 9.13e-155 |
| make_stone_sword | 3 | 616 | 93 | 613 | 0.1517 | 101.46 | 7.78e-152 |
| make_wood_pickaxe | 3 | 903 | 19 | 35 | 0.5429 | 324.74 | 6.88e-44 |
| make_wood_sword | 3 | 903 | 14 | 35 | 0.4000 | 243.23 | 2.38e-30 |
| place_furnace | 0 | 51 | 89 | 412 | 0.2160 | 132.17 | 6.45e-157 |
| place_plant | 3 | 938 | 477 | 519 | 0.9191 | 543.19 | 0.0 |
| place_stone | 3 | 273 | 473 | 2734 | 0.1730 | 104.34 | 0.0 |
| place_table | 3 | 36 | 480 | 1507 | 0.3185 | 187.87 | 0.0 |
| wake_up | 1 | 583 | 149 | 400 | 0.3725 | 233.24 | 2.74e-304 |

(Note: combat events have weak detector codes — best P(a|c) is only 0.066 for
defeat_skeleton and 0.186 for defeat_zombie.)

---

## 4. PROBES (canonical: probes_hud, 7 representations)

Sources: `results/probes_hud/probe_metrics.csv` (reproduced verbatim below);
exact comparisons from `scratch/probes_hud/probe_metrics.json`;
n from `scratch/probes_hud/probe_meta.json`.

From `probe_meta.json`: **n_samples = 81,919** (150 episodes); split per
`probe_metrics.json`: n_train = 64,463, n_test = 17,456.
Representations (7): raw_codes, frame_emb, wm_latents_emb, wm_input,
wm_block_1, wm_block_2, wm_block_3. ("wm_input" is actually the OUTPUT of
transformer block 0 — hooks capture block exits; true pre-transformer reps are
frame_emb and wm_latents_emb.)

Metric: AUROC for binary labels; accuracy for the multiclass `action_taken` row.
Empty cells = label never positive (4 never/rare-unlocked achievements).

Full table (verbatim from `results/probes_hud/probe_metrics.csv`):

```
label,raw_codes,frame_emb,wm_latents_emb,wm_input,wm_block_1,wm_block_2,wm_block_3
action_taken,0.3555,0.4997,0.6083,0.9802,0.9738,0.9588,0.9559
reward_in_next_5,0.5649,0.7892,0.6078,0.7658,0.7875,0.8060,0.8084
reward_now,0.8144,0.8383,0.9596,0.9731,0.9772,0.9763,0.9761
ach_just[collect_coal],0.9813,0.9981,0.9677,0.9857,1.0000,1.0000,1.0000
ach_just[collect_diamond],,,,,,,
ach_just[collect_drink],0.8977,0.9976,0.9384,0.9763,0.9745,0.9470,0.9591
ach_just[collect_iron],0.9999,0.9996,1.0000,1.0000,1.0000,1.0000,1.0000
ach_just[collect_sapling],1.0000,0.9881,1.0000,1.0000,1.0000,1.0000,1.0000
ach_just[collect_stone],0.9861,0.9918,0.9749,0.9985,0.9989,0.9996,0.9995
ach_just[collect_wood],0.9922,0.9999,0.9831,0.9995,0.9998,1.0000,1.0000
ach_just[defeat_skeleton],0.8642,0.9675,0.8229,0.9481,0.9721,0.9770,0.9655
ach_just[defeat_zombie],0.8504,0.9789,0.8624,0.9948,0.9862,0.9947,0.9923
ach_just[eat_cow],0.9472,0.9939,0.9246,0.9980,0.9993,0.9996,0.9998
ach_just[eat_plant],,,,,,,
ach_just[make_iron_pickaxe],,,,,,,
ach_just[make_iron_sword],,,,,,,
ach_just[make_stone_pickaxe],0.9873,0.9989,0.9944,0.9762,0.9939,0.9862,0.9469
ach_just[make_stone_sword],0.9573,0.9993,0.9134,0.9998,1.0000,1.0000,1.0000
ach_just[make_wood_pickaxe],0.9880,0.9989,0.9885,0.9999,1.0000,1.0000,1.0000
ach_just[make_wood_sword],0.9765,0.9960,0.9907,0.9999,1.0000,1.0000,1.0000
ach_just[place_furnace],0.9702,0.9908,0.9279,0.9992,0.9978,0.9991,0.9993
ach_just[place_plant],0.9922,1.0000,0.9845,1.0000,1.0000,1.0000,1.0000
ach_just[place_stone],0.9740,0.9785,0.9604,0.9926,0.9974,0.9950,0.9853
ach_just[place_table],0.9922,0.9934,0.9872,0.9996,0.9988,0.9976,0.9928
ach_just[wake_up],0.9695,0.9998,0.9727,0.9987,0.9979,0.9981,0.9933
ach_cum[collect_coal],0.6228,1.0000,0.6592,0.9433,0.9526,0.9504,0.9536
ach_cum[collect_diamond],,,,,,,
ach_cum[collect_drink],0.6132,0.9652,0.6556,0.9042,0.9104,0.9150,0.9196
ach_cum[collect_iron],0.6115,1.0000,0.6569,0.9077,0.9353,0.9244,0.9313
ach_cum[collect_sapling],0.9142,0.9993,0.9446,1.0000,1.0000,1.0000,1.0000
ach_cum[collect_stone],0.6017,0.9986,0.6794,0.9862,0.9842,0.9799,0.9819
ach_cum[collect_wood],0.7992,0.9996,0.8478,0.9983,0.9978,0.9980,0.9981
ach_cum[defeat_skeleton],0.6009,0.8519,0.6219,0.8424,0.8431,0.8394,0.8383
ach_cum[defeat_zombie],0.6002,0.9726,0.6570,0.9557,0.9515,0.9531,0.9538
ach_cum[eat_cow],0.6589,0.9842,0.7052,0.9710,0.9720,0.9717,0.9716
ach_cum[eat_plant],,,,,,,
ach_cum[make_iron_pickaxe],,,,,,,
ach_cum[make_iron_sword],,,,,,,
ach_cum[make_stone_pickaxe],0.6088,0.9999,0.6697,0.9888,0.9822,0.9768,0.9808
ach_cum[make_stone_sword],0.6144,1.0000,0.6686,0.9891,0.9799,0.9748,0.9798
ach_cum[make_wood_pickaxe],0.6075,1.0000,0.6825,0.9939,0.9921,0.9868,0.9886
ach_cum[make_wood_sword],0.6000,0.9998,0.6438,0.9742,0.9576,0.9490,0.9561
ach_cum[place_furnace],0.5918,0.9658,0.6436,0.9544,0.9504,0.9442,0.9467
ach_cum[place_plant],0.9195,1.0000,0.9486,1.0000,1.0000,1.0000,1.0000
ach_cum[place_stone],0.6008,0.9975,0.6748,0.9845,0.9807,0.9779,0.9803
ach_cum[place_table],0.6785,0.9962,0.7339,0.9831,0.9800,0.9801,0.9807
ach_cum[wake_up],0.5744,0.9164,0.6047,0.8577,0.8820,0.8539,0.8503
```

HUD-shortcut count (exact comparison using full-precision AUROCs from
`scratch/probes_hud/probe_metrics.json`, NOT the rounded CSV):
**ach_cum concepts with frame_emb >= wm_block_1: 16 of 18.**
The same 16/18 holds for "frame_emb >= ALL transformer blocks (wm_input..block_3)".
The only two exceptions (both with all AUROCs > 0.999):
- ach_cum[collect_sapling]: frame_emb 0.999334 < wm_block_1 0.999997
- ach_cum[place_plant]: frame_emb 0.999993 < wm_block_1 1.000000
(HANDOFF.md quotes "15/18"; the exact JSON comparison gives 16/18 — use 16/18.)

Anticipation (reward_in_next_5) is the one label that improves with depth and
where frame_emb is not best: 0.7892 (frame_emb) vs 0.7658 (wm_input/block-0 out)
-> 0.7875 (block 1) -> 0.8060 (block 2) -> 0.8084 (block 3).

---

## 5. CAVs

Sources: `results/cavs/cav_metrics.json`; cosines recomputed from
`scratch/cavs/cavs.npz` (`w_wm_block_1`, shape (44 concepts, 512)).

`cav_metrics.json` contains 220 rows = 44 concepts x 5 reps (raw_codes,
wm_input, wm_block_1, wm_block_2, wm_block_3). 40 rows have auroc = null
(8 concepts never/too-rarely positive: just/cum of collect_diamond, eat_plant,
make_iron_pickaxe, make_iron_sword), leaving 36 valid concepts per rep.

CAVs with held-out AUROC >= 0.85, per rep (out of 36 valid):

| rep | n(auroc>=0.85) | n(auroc>=0.90) |
|---|---|---|
| raw_codes | 20 | 17 |
| wm_input | 35 | 34 |
| wm_block_1 | 35 | 34 |
| wm_block_2 | 35 | 34 |
| wm_block_3 | 35 | 34 |

Cosine geometry at wm_block_1, restricted to the 34 concepts with
auroc >= 0.90 at wm_block_1 (rows of `w_wm_block_1` L2-normalised, all
561 pairwise cosines):

Top-5 most-aligned pairs:

| pair | cos |
|---|---|
| cum[collect_stone] ~ cum[place_stone] | **+0.9525** |
| cum[collect_stone] ~ cum[make_wood_pickaxe] | +0.8744 |
| cum[make_wood_pickaxe] ~ cum[place_stone] | +0.8323 |
| cum[collect_sapling] ~ cum[place_plant] | +0.8322 |
| cum[make_stone_pickaxe] ~ cum[make_stone_sword] | +0.7931 |

5 most negative pairs:

| pair | cos |
|---|---|
| cum[place_plant] ~ just[collect_sapling] | -0.3010 |
| cum[collect_drink] ~ just[collect_drink] | -0.2554 |
| cum[place_table] ~ just[collect_wood] | -0.2528 |
| cum[make_wood_pickaxe] ~ just[place_table] | -0.2044 |
| cum[collect_wood] ~ just[collect_wood] | -0.1971 |

Double-checked famous pair: **cos(cum[collect_stone], cum[place_stone]) =
0.952521** (rounds to +0.95 as in HANDOFF.md). Note the just[X] ~ cum[X]
cosines are mildly negative (~ -0.2 to -0.3): event direction is roughly
orthogonal-to-slightly-anti-aligned with the corresponding state direction.

---

## 6. SAE

Sources: `results/sae/features.json`; FVU/time from
`runs/train-sae-21836854.out`.

- layer: **wm_block_1** (summary activations); dictionary **n_features = 2048**,
  TopK **k = 16**; trained on N = 81,919 samples of dim D = 512.
- FVU (final epoch line, verbatim): `epoch  80/80  train MSE=0.18802  val
  MSE=0.23307  FVU=0.043  alive_features=2048/2048` -> **val FVU = 0.043**,
  **0 dead features**.
- training time: **95.69 s** (job 21836854, Fri Jun 5 02:08:50 -> 02:10:25 2026).
- density (fraction of samples on which feature is active), over all 2048
  features: mean 0.007812, median 0.004962, min 0.000378, max 0.467743;
  percentiles 5/25/50/75/95 = 0.001636 / 0.002918 / 0.004962 / 0.008435 /
  0.018386; features with density 0: **0**.

Best feature per achievement (max `best_ach.lift` over features whose
best_ach is that achievement). All 18 achievements that appear have best
lift >= 10, so all are listed ("P" is the `best_ach.p` field of features.json):

| achievement | feature | lift | P | density | best_cav (cos) | best_code |
|---|---|---|---|---|---|---|
| collect_iron | f1820 | 623.54 | 0.784 | 0.00607 | just[collect_iron] (+0.411) | (s3,c75) |
| collect_coal | f1742 | 589.35 | 1.000 | 0.00526 | just[collect_coal] (+0.577) | (s3,c578) |
| make_wood_pickaxe | f1451 | 548.13 | 0.9903 | 0.00499 | just[make_wood_pickaxe] (+0.459) | (s3,c36) |
| place_plant | f266 | 546.13 | 1.000 | 0.00400 | just[place_plant] (+0.327) | (s3,c938) |
| place_table | f677 | 527.40 | 0.9528 | 0.00518 | just[place_table] (+0.455) | (s3,c36) |
| make_stone_pickaxe | f872 | 521.51 | 0.8849 | 0.00679 | just[make_stone_pickaxe] (+0.481) | (s3,c616) |
| collect_sapling | f615 | 518.48 | 0.9494 | 0.00383 | just[collect_sapling] (+0.400) | (s2,c707) |
| place_stone | f431 | 478.49 | 0.8469 | 0.00479 | just[place_stone] (+0.340) | (s3,c273) |
| make_wood_sword | f372 | 444.21 | 0.770 | 0.00486 | just[make_wood_sword] (+0.230) | (s3,c36) |
| wake_up | f207 | 436.68 | 0.7356 | 0.00421 | just[wake_up] (+0.237) | (s0,c783) |
| place_furnace | f1175 | 389.40 | 0.675 | 0.00391 | just[place_furnace] (+0.184) | (s3,c273) |
| make_stone_sword | f316 | 377.81 | 0.6042 | 0.00466 | just[make_stone_sword] (+0.246) | (s3,c616) |
| eat_cow | f759 | 277.85 | 0.5054 | 0.00454 | just[eat_cow] (+0.175) | (s2,c648) |
| collect_drink | f561 | 207.77 | 0.3652 | 0.01119 | just[collect_drink] (+0.201) | (s2,c991) |
| collect_wood | f187 | 172.58 | 0.316 | 0.01220 | just[collect_wood] (+0.251) | (s3,c940) |
| defeat_zombie | f792 | 166.16 | 0.2759 | 0.01129 | just[defeat_zombie] (+0.192) | (s0,c941) |
| collect_stone | f86 | 149.83 | 0.2689 | 0.01032 | just[collect_stone] (+0.170) | (s3,c991) |
| defeat_skeleton | f1483 | 120.25 | 0.1952 | 0.01023 | just[defeat_skeleton] (+0.098) | (s2,c648) |

Polysemantic-code split — features whose `best_code` is (slot 3, code 36).
CORRECTION TO HANDOFF: there are **8** such features, not 3. The three
headline features that split (s3,c36) into separate concepts (the ones quoted
in HANDOFF.md / the paper claim) are marked *:

| feature | best_ach | lift | P | density | best_cav (cos) |
|---|---|---|---|---|---|
| f677 * | place_table | 527.40 | 0.9528 | 0.00518 | just[place_table] (+0.455) |
| f1451 * | make_wood_pickaxe | 548.13 | 0.9903 | 0.00499 | just[make_wood_pickaxe] (+0.459) |
| f372 * | make_wood_sword | 444.21 | 0.770 | 0.00486 | just[make_wood_sword] (+0.230) |
| f2004 | make_wood_sword | 295.60 | 0.5124 | 0.01182 | just[make_wood_sword] (+0.269) |
| f639 | make_wood_pickaxe | 272.31 | 0.492 | 0.00913 | just[make_stone_sword] (+0.208) |
| f383 | make_wood_pickaxe | 404.72 | 0.7312 | 0.00453 | cum[defeat_skeleton] (-0.133) |
| f2017 | make_wood_pickaxe | 66.02 | 0.1193 | 0.00532 | just[place_plant] (+0.103) |
| f1834 | collect_wood | 25.54 | 0.0468 | 0.01354 | cum[make_stone_pickaxe] (-0.156) |

The * trio are each the single best feature for their achievement, i.e. the SAE
assigns place_table / make_wood_pickaxe / make_wood_sword to distinct features
even though their dominant code is the same (s3,c36).

Triply-corroborated features (|best_cav.cos| > 0.4 AND best_ach.lift > 10):
**6 features**, of which 5 have matching achievement/CAV names:

| feature | best_ach (lift) | best_cav (cos) | best_code | names match |
|---|---|---|---|---|
| f872 | make_stone_pickaxe (521.51) | just[make_stone_pickaxe] (+0.4808) | (s3,c616) | yes |
| f1820 | collect_iron (623.54) | just[collect_iron] (+0.4107) | (s3,c75) | yes |
| f1742 | collect_coal (589.35) | just[collect_coal] (+0.5767) | (s3,c578) | yes |
| f677 | place_table (527.40) | just[place_table] (+0.4550) | (s3,c36) | yes |
| f1451 | make_wood_pickaxe (548.13) | just[make_wood_pickaxe] (+0.4593) | (s3,c36) | yes |
| f33 | collect_sapling (58.72) | cum[place_plant] (-0.4287) | (s0,c751) | no (related concepts) |

---

## 7. ONE-STEP ABLATION

Sources: `results/ablation_gap0/causal_effects.json` and
`results/ablation_gap3/causal_effects.json` (72 records each: 18 concepts x
{unlock, ordinary} x {feat, rand}). Effect = change in NLL (nats) of the real
next codes after projecting the direction out of block-1 output at the moment
step. mean with 95% CI [ci_lo, ci_hi] as stored in the JSON.
n = 80 per cell except unlock cells for collect_sapling (n=26) and
place_plant (n=29). (An ordinary/rand control also exists in the files.)

### gap = 0 (ablate at the prediction step)

| concept | unlock,feat mean [95% CI] (n) | ordinary,feat mean [95% CI] (n) | unlock,rand mean [95% CI] (n) |
|---|---|---|---|
| collect_coal | **+2.406 [+1.954, +2.815]** (80) | -0.050 [-0.113, +0.002] (80) | -0.019 [-0.121, +0.059] (80) |
| collect_drink | -0.334 [-0.530, -0.123] (80) | -0.113 [-0.228, -0.011] (80) | -0.094 [-0.177, -0.007] (80) |
| collect_iron | -0.341 [-0.454, -0.236] (80) | +0.179 [+0.067, +0.320] (80) | +0.021 [-0.050, +0.093] (80) |
| collect_sapling | **+1.733 [+1.302, +2.116]** (26) | -0.030 [-0.083, +0.019] (80) | -0.042 [-0.130, +0.043] (26) |
| collect_stone | +0.047 [-0.035, +0.151] (80) | +0.112 [+0.049, +0.175] (80) | +0.006 [-0.057, +0.071] (80) |
| collect_wood | +0.011 [-0.036, +0.052] (80) | +0.051 [-0.021, +0.111] (80) | -0.073 [-0.160, -0.001] (80) |
| defeat_skeleton | **+2.202 [+1.483, +3.032]** (80) | -0.391 [-0.541, -0.226] (80) | -0.009 [-0.088, +0.064] (80) |
| defeat_zombie | +0.717 [+0.435, +1.033] (80) | +0.262 [+0.079, +0.469] (80) | +0.008 [-0.066, +0.100] (80) |
| eat_cow | -0.102 [-0.190, -0.032] (80) | +0.104 [+0.010, +0.202] (80) | +0.007 [-0.084, +0.100] (80) |
| make_stone_pickaxe | +0.436 [+0.269, +0.647] (80) | -0.078 [-0.159, -0.002] (80) | -0.006 [-0.101, +0.094] (80) |
| make_stone_sword | +0.222 [+0.128, +0.334] (80) | +0.045 [-0.038, +0.127] (80) | +0.020 [-0.074, +0.121] (80) |
| make_wood_pickaxe | -0.117 [-0.182, -0.049] (80) | -0.035 [-0.099, +0.022] (80) | -0.080 [-0.178, +0.006] (80) |
| make_wood_sword | +0.004 [-0.071, +0.078] (80) | -0.064 [-0.114, -0.022] (80) | +0.074 [-0.034, +0.199] (80) |
| place_furnace | +0.133 [-0.003, +0.289] (80) | +0.780 [+0.523, +1.073] (80) | -0.058 [-0.108, -0.012] (80) |
| place_plant | +0.322 [+0.159, +0.503] (29) | +0.299 [+0.079, +0.512] (80) | +0.003 [-0.057, +0.059] (29) |
| place_stone | -0.083 [-0.172, +0.032] (80) | +0.176 [+0.091, +0.271] (80) | +0.049 [-0.021, +0.123] (80) |
| place_table | -0.030 [-0.285, +0.229] (80) | -0.241 [-0.398, -0.077] (80) | +0.020 [-0.041, +0.091] (80) |
| wake_up | **+4.855 [+4.381, +5.381]** (80) | +0.077 [-0.064, +0.209] (80) | +0.037 [-0.066, +0.129] (80) |

Headline: max damage = **wake_up +4.855 nats** [4.381, 5.381]; both controls
(ordinary moments and random direction) are ~0 everywhere (|mean| <= 0.78, most
<= 0.11). Note 4 concepts show significant negative (helpful) unlock effects
(collect_drink, collect_iron, eat_cow, make_wood_pickaxe), all small (<= 0.34).

### gap = 3 (ablate 3 steps before the prediction step)

Max |unlock,feat| effect over all 18 concepts: **place_stone, mean = +0.0227**
[+0.0137, +0.0320] (n=80). Every other concept is smaller in magnitude
(second largest: wake_up -0.0168 [-0.0281, -0.0078]). I.e. the gap-0 effects
(up to 4.86 nats) collapse by >2 orders of magnitude at gap 3 — the model
recovers via the frame path.

---

## 8. IMAGINATION (sustained ablation during 12-step dreams)

Source: `results/imagination/imagine_effects.json` (72 records: 18 concepts x
4 conditions {baseline, sae, cav, random}; n = 20 dreams per cell).
p_first = P(detector code at the first imagined step where it appeared in the
real rollout); p_within = P(detector code anywhere within the horizon);
[lo, hi] = 95% CI as stored in the file.

| concept | condition | p_first [CI] | p_within [CI] |
|---|---|---|---|
| make_wood_pickaxe | baseline | 0.80 [0.60, 0.95] | 0.80 [0.60, 0.95] |
| make_wood_pickaxe | sae | 0.80 [0.60, 0.95] | 0.80 [0.60, 0.95] |
| make_wood_pickaxe | cav | 0.80 [0.60, 0.95] | 0.80 [0.60, 0.95] |
| make_wood_pickaxe | random | 0.80 [0.60, 0.95] | 0.80 [0.60, 0.95] |
| collect_iron | baseline | 0.90 [0.75, 1.00] | 1.00 [1.00, 1.00] |
| collect_iron | sae | 0.75 [0.55, 0.90] | 0.85 [0.70, 1.00] |
| collect_iron | cav | 0.85 [0.70, 1.00] | 0.95 [0.85, 1.00] |
| collect_iron | random | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |
| collect_sapling | baseline | 0.05 [0.00, 0.15] | 0.60 [0.40, 0.80] |
| collect_sapling | sae | 0.05 [0.00, 0.15] | 0.40 [0.20, 0.60] |
| collect_sapling | cav | 0.10 [0.00, 0.25] | 0.50 [0.30, 0.70] |
| collect_sapling | random | 0.05 [0.00, 0.15] | 0.70 [0.50, 0.90] |
| collect_coal | baseline | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| collect_coal | sae | 0.70 [0.50, 0.90] | 0.70 [0.50, 0.90] |
| collect_coal | cav | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| collect_coal | random | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| eat_cow | baseline | 0.70 [0.50, 0.90] | 0.75 [0.55, 0.90] |
| eat_cow | sae | 0.75 [0.55, 0.90] | 0.75 [0.55, 0.90] |
| eat_cow | cav | 0.65 [0.45, 0.85] | 0.75 [0.55, 0.90] |
| eat_cow | random | 0.75 [0.55, 0.90] | 0.75 [0.55, 0.90] |
| make_wood_sword | baseline | 0.90 [0.75, 1.00] | 0.90 [0.75, 1.00] |
| make_wood_sword | sae | 0.90 [0.75, 1.00] | 0.90 [0.75, 1.00] |
| make_wood_sword | cav | 0.90 [0.75, 1.00] | 0.90 [0.75, 1.00] |
| make_wood_sword | random | 0.90 [0.75, 1.00] | 0.90 [0.75, 1.00] |
| place_furnace | baseline | 0.45 [0.25, 0.65] | 0.45 [0.25, 0.65] |
| place_furnace | sae | 0.45 [0.25, 0.65] | 0.45 [0.25, 0.65] |
| place_furnace | cav | 0.45 [0.25, 0.65] | 0.45 [0.25, 0.65] |
| place_furnace | random | 0.45 [0.25, 0.65] | 0.45 [0.25, 0.65] |
| make_stone_pickaxe | baseline | 0.75 [0.55, 0.95] | 0.80 [0.60, 0.95] |
| make_stone_pickaxe | sae | 0.55 [0.35, 0.75] | 0.70 [0.50, 0.90] |
| make_stone_pickaxe | cav | 0.75 [0.55, 0.95] | 0.80 [0.60, 0.95] |
| make_stone_pickaxe | random | 0.75 [0.55, 0.95] | 0.80 [0.60, 0.95] |
| wake_up | baseline | 0.30 [0.10, 0.50] | 0.30 [0.10, 0.50] |
| wake_up | sae | 0.50 [0.30, 0.70] | 0.50 [0.30, 0.70] |
| wake_up | cav | 0.35 [0.15, 0.60] | 0.35 [0.15, 0.60] |
| wake_up | random | 0.30 [0.10, 0.50] | 0.30 [0.10, 0.50] |
| defeat_zombie | baseline | 0.15 [0.00, 0.30] | 0.15 [0.00, 0.30] |
| defeat_zombie | sae | 0.15 [0.00, 0.30] | 0.15 [0.00, 0.30] |
| defeat_zombie | cav | 0.15 [0.00, 0.30] | 0.20 [0.05, 0.40] |
| defeat_zombie | random | 0.15 [0.00, 0.30] | 0.20 [0.05, 0.40] |
| collect_drink | baseline | 0.30 [0.10, 0.50] | 0.60 [0.40, 0.80] |
| collect_drink | sae | 0.30 [0.10, 0.50] | 0.60 [0.40, 0.80] |
| collect_drink | cav | 0.30 [0.10, 0.50] | 0.65 [0.45, 0.85] |
| collect_drink | random | 0.25 [0.10, 0.45] | 0.55 [0.35, 0.75] |
| collect_wood | baseline | **1.00 [1.00, 1.00]** | **1.00 [1.00, 1.00]** |
| collect_wood | sae | **0.00 [0.00, 0.00]** | **0.10 [0.00, 0.25]** |
| collect_wood | cav | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| collect_wood | random | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| place_table | baseline | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| place_table | sae | 0.90 [0.75, 1.00] | 0.90 [0.75, 1.00] |
| place_table | cav | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| place_table | random | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| place_plant | baseline | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |
| place_plant | sae | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| place_plant | cav | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| place_plant | random | 0.95 [0.85, 1.00] | 0.95 [0.85, 1.00] |
| place_stone | baseline | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |
| place_stone | sae | 0.70 [0.50, 0.90] | 1.00 [1.00, 1.00] |
| place_stone | cav | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| place_stone | random | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| defeat_skeleton | baseline | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| defeat_skeleton | sae | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| defeat_skeleton | cav | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| defeat_skeleton | random | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| make_stone_sword | baseline | 0.75 [0.55, 0.90] | 0.75 [0.55, 0.90] |
| make_stone_sword | sae | 0.45 [0.25, 0.65] | 0.45 [0.25, 0.65] |
| make_stone_sword | cav | 0.80 [0.60, 0.95] | 0.80 [0.60, 0.95] |
| make_stone_sword | random | 0.80 [0.60, 0.95] | 0.80 [0.60, 0.95] |
| collect_stone | baseline | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |
| collect_stone | sae | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |
| collect_stone | cav | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |
| collect_stone | random | 0.95 [0.85, 1.00] | 1.00 [1.00, 1.00] |

(Exact CI endpoints that the JSON stores with more digits: collect_iron sae
p_first_hi = 0.90125; defeat_zombie baseline/cav/random p_first_hi = 0.30125.
All other values are exact multiples of 0.05 as shown.)

Headline numbers: collect_wood baseline p_first/p_within = 1.00/1.00 -> SAE
ablation 0.00/0.10, while CAV and random both stay at 1.00/1.00.
Moderate SAE effects: make_stone_sword p_within 0.75 -> 0.45; collect_coal
0.95 -> 0.70; collect_iron 1.00 -> 0.85 (p_within). CAV ablation is inert for
every concept (max change vs baseline: |Δp_within| <= 0.10).
Known oddities: wake_up increases under SAE ablation (0.30 -> 0.50);
defeat_skeleton detector never fires in dreams in any condition (0.00 across
the board); defeat_* outcome metric is weak (detector P(a|c) ~ 0.07).

---

## 9. CAUSAL TRACE (activation patching)

Source: `results/causal_trace/trace_results.json`. 4 concepts, each with
n = 24 trials, grid shape = (3 token-groups [frame, act, latents]) x
(3 blocks) x (21 window steps); restoration = (lp_patched - lp_corrupt) /
(lp_clean - lp_corrupt), clamped to [-1, 2] (per `analysis/causal_trace.py`).
Step 20 is the final (unlock) step. "mean(steps<15)" = mean restoration over
the 15 earliest history steps, all blocks of that group.

| concept | n | lp_clean | lp_corrupt | gap (nats) |
|---|---|---|---|---|
| wake_up | 24 | -23.796 | -49.948 | 26.151 |
| collect_coal | 24 | -18.061 | -36.542 | 18.481 |
| defeat_skeleton | 24 | -7.956 | -25.289 | 17.333 |
| collect_sapling | 24 | -23.675 | -38.761 | 15.086 |

Per token-group: max restoration (and where), history mean, and final-step
(step 20) restoration per block:

wake_up:
- frame: max 0.1357 (block 0, step 20); mean steps<15 = 0.0022; final step b0/b1/b2 = 0.1357 / -0.0128 / 0.0000
- act: max **0.5837 (block 2, step 20)**; mean steps<15 = -0.0001; final step = 0.4841 / 0.4962 / 0.5837
- latents: max 0.4163 (block 2, step 20); mean steps<15 = 0.0014; final step = 0.2195 / 0.3763 / 0.4163

collect_coal:
- frame: max 0.1179 (block 1, step 20); mean steps<15 = -0.0243; final step = 0.0791 / 0.1179 / 0.0000
- act: max 0.4818 (block 2, step 20); mean steps<15 = -0.0172; final step = 0.4784 / 0.4597 / 0.4818
- latents: max **0.5005 (block 1, step 20)**; mean steps<15 = -0.0217; final step = 0.3439 / 0.5005 / 0.3932

defeat_skeleton:
- frame: max 0.1703 (block 1, step 2); mean steps<15 = 0.0332; final step = 0.0386 / -0.0910 / 0.0000
- act: max **0.4555 (block 2, step 20)**; mean steps<15 = 0.0136; final step = 0.2477 / 0.2763 / 0.4555
- latents: max 0.3605 (block 0, step 20); mean steps<15 = 0.0418; final step = 0.3605 / 0.3395 / 0.2945

collect_sapling:
- frame: max 0.2317 (block 0, step 20); mean steps<15 = -0.0187; final step = 0.2317 / 0.0940 / 0.0000
- act: max 0.4609 (block 2, step 20); mean steps<15 = 0.0025; final step = 0.3666 / 0.4051 / 0.4609
- latents: max **0.5391 (block 2, step 20)**; mean steps<15 = 0.0028; final step = 0.2931 / 0.3487 / 0.5391

Summary facts: in 11 of 12 (concept x group) cells the maximum restoration sits
at the FINAL step (only exception: defeat_skeleton frame, block 1 step 2,
0.1703). All history means (steps<15) are |mean| <= 0.042 ~ 0. Action token at
block 2, final step restores 0.4555–0.5837 across all 4 concepts (it is the
single largest cell for wake_up and defeat_skeleton; latents is largest for
collect_coal, 0.5005, and collect_sapling, 0.5391); frame final-step max is
only 0.1179–0.2317. (Frame-at-block-2 is identically 0.0000 by construction: the
final block's frame positions are not read by the code head.)

---

## 10. MISC / PARAMETERS

- SAE training: **95.69 s** total job execution (job 21836854; start Fri Jun 5
  02:08:50 2026, end 02:10:25 2026). Source: `runs/train-sae-21836854.out`.
  SAE config (from `runs/train_sae.sbatch` + `results/sae/features.json`):
  layer wm_block_1, 2048 features, k=16, 80 epochs.
- Probe dataset: **n_samples = 81,919** over 150 episodes (source:
  `scratch/probes_hud/probe_meta.json`); train/test split 64,463 / 17,456
  (source: `scratch/probes_hud/probe_metrics.json`).
- Imagination params (source: `runs/imagine_ablation.sbatch` +
  `analysis/imagine_ablation.py` defaults): **burn-in 8, horizon 12,
  n_moments 20 per concept x condition, target block 1, seed 42**.
  (Horizon limited to 12 by the 21-slot KV cache: 8 burn-in + 12 + 1.)
- One-step ablation params (source: `runs/ablate_features.sbatch` +
  `analysis/ablate_features.py` defaults): **window 21, strength 1.0,
  n-trials-per-set 80, target block 1, seed 42**, gaps 0 and 3.
- Causal-trace params (source: `runs/causal_trace.sbatch`): window 21,
  n-trials 24 (4 concepts traced).
- Codebook scan wall-time: 1683.8 s (`results/meta.json`, `seconds_elapsed`).
- Agent: Δ-IRIS, Crafter, 1000 epochs, seed 0, ~8.7 days on one H200
  (job chain 21531393 -> 21577143 -> 21690008; source: HANDOFF.md — job-level
  facts not independently re-verified here).

## Discrepancies vs HANDOFF.md found during verification

1. HUD count: HANDOFF says frame_emb matches/beats blocks on "15/18" ach_cum
   concepts; exact recomputation from probe_metrics.json gives **16/18**
   (exceptions: collect_sapling, place_plant, both with AUROC > 0.999 anyway).
2. (s3,c36) SAE features: HANDOFF implies 3; **8 features** have
   best_code=(s3,c36). The headline split trio f677/f1451/f372 is correct and
   each is its achievement's best feature.
3. Imagination collect_coal: HANDOFF quotes ".95→.70" (p_within) — confirmed;
   collect_iron "1.0→.85" is p_within (p_first is 0.90→0.75); make_stone_sword
   ".75→.45" confirmed.
4. Mean unlocked achievements per episode is 17.10 by the unlock-count table;
   the mean *return* is 16.20 (return also includes -0.1/+0.1 health rewards),
   HANDOFF's "~16/22 achievements per episode" conflates the two slightly.
