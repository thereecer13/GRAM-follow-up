# Measure this machine, then run the real model end to end on 1% of the data.
set -e
python scripts/benchmark.py --price-per-hour 0.89
python scripts/run_set.py configs/smoke
python scripts/make_results.py runs/smoke --out results/smoke --check
