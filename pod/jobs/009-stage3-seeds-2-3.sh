# Stage 3, seeds 2 and 3 (seed 1's finished runs are skipped), then tables over all three seeds.
set -e
python scripts/run_set.py configs/stage3
python scripts/make_results.py runs/stage3 --out results/stage3 --check-new
du -sh /workspace/GRAM-follow-up/runs/*
df -h /workspace | tail -1
