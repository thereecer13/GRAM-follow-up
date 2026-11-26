# Stage 3's code path (mixed batches, unlabeled data, compiled) on 1% of the data, and a disk report.
set -e
df -h /workspace | tail -1
du -sh /workspace/GRAM-follow-up/runs/* /workspace/GRAM-follow-up/data 2>/dev/null
python scripts/run_set.py configs/stage3_smoke
python scripts/make_results.py runs/stage3_smoke --out results/stage3_smoke --check
