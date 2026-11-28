# Seeds 2 and 3 of Stage 3 were stopped early (decision of 2026-11-28). Rebuild the tables from the
# runs that finished: all of seed 1, plus the baselines and filtering at fraction 0.01 of seeds 2 and 3.
set -e
python scripts/make_results.py runs/stage3 --out results/stage3 --check-new
du -sh /workspace/GRAM-follow-up/runs/*
df -h /workspace | tail -1
