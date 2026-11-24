# v2 GRPO round 1 (2026-11-16)

- Data: processed_v2 (70/10/20), GRPO on train 18,574 x (2 Student1 x 2 Student2) rollouts from the v2 SFT adapters.
- Rewards: individual (with start_time) + distribution, LOO Q, weights sequence 0.82 / trip_mode 1.04 / first_departure 0.43
  (50:50 within-group spread, v2/weights.json). Generation T 0.7 / top_p 0.9.
- GRPO: lr 1e-5, 1 epoch (Student1 1,153 / Student2 2,312 steps), ref penalty 0.02, batch 2 x accum 16.
  Final |log ratio| S1 0.037 (clip 3.5%), S2 0.016 (clip 1.5%).
- Files: adapters v2/grpo/student{1,2}/final; outputs v2/post_rl_{test,validation}; Pre-RL v2/pre_rl_*;
  metrics v2/analysis/{pre_rl,post_rl}_{split}[_noref]/metrics.json; bootstrap v2/analysis/bootstrap_{split}.json;
  figures v2/analysis/distributions_{split}.

Pre-RL -> Round 1, both after refinement; paired bootstrap 500, * = 95% CI excludes 0.

| metric | test Pre | test R1 | test diff | validation diff |
|---|---|---|---|---|
| exact sequence | .248 | .254 | +.006 | +.013 |
| LCS F1 | .697 | .698 | +.002 | +.007 |
| location (dong) | .758 | .757 | -.001 | -.000 |
| dwell +-30 min | .227 | .222 | -.006 | +.001 |
| trip mode | .360 | .346 | -.014 | -.032 * |
| JSD sequence pattern | .054 | .057 | +.002 | -.002 |
| JSD activity count | .046 | .048 | +.001 | -.002 |
| JSD activity type | .020 | .021 | +.001 | -.006 * |
| JSD trip mode | .036 | .024 | -.012 * | -.017 * |
| JSD dwell | .092 | .093 | +.002 | +.003 |
| JSD first departure | .058 | .046 | -.012 * | -.015 * |
| JSD travel time | .081 | .085 | +.004 | +.018 * |
