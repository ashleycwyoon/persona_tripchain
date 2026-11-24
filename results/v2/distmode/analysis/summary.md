# v2 GRPO round 1, distance-conditioned trip-mode reward (2026-11-17)
Student1 = round 1 (same updates). Student2 retrained from SFT with trip_mode cells band x occupation x age x gender (prior 100), weight 0.50.
Paired bootstrap (500, person level), * = 95% CI excludes 0.

| test | Pre-RL | Round1 | DistMode |
|---|---|---|---|
| trip mode acc (LCS) | .360 | .346 | .355 (vs R1 +.009, vs Pre -.005, both ns) |
| JSD trip mode | .036 | .024 | .028 (vs Pre -.008*, vs R1 +.004*) |
| subway share (GT .181) | .222 | .182 | .218 |
| walk share (GT .351) | .402 | .392 | .389 |

validation: trip mode acc .351 / .319 / .340 (vs R1 +.021*, vs Pre -.011 ns); JSD trip mode .042 / .026 / .029 (vs Pre -.013*).

Per GT mode accuracy, test (Pre / R1 / DistMode): walk .538/.502/.528, subway .377/.312/.362, car_driver .434/.464/.456, bus .153/.183/.138.
Details: per_mode_band_accuracy.json, bootstrap_*.json, distributions_*/.
