# Stage 3, seed 1: a baseline, filtering and GRAM at four labeled fractions, and the same-steps control.
set -e
python scripts/run_set.py configs/stage3
python scripts/make_results.py runs/stage3 --out results/stage3 --check
du -sh /workspace/GRAM-follow-up/runs/*
