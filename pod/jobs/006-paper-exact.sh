# Stage 2: the paper_exact set (baseline, core-only filtering, GRAM), then tables and the comparison with the paper.
set -e
python scripts/run_set.py configs/paper_exact
python scripts/make_results.py runs/paper_exact --out results/paper_exact --check
python scripts/compare_with_paper.py results/paper_exact
