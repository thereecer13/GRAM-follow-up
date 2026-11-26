# Stage 2: the code_exact set (baseline, core-only filtering, GRAM), then tables and the comparison with the paper.
set -e
python scripts/run_set.py configs/code_exact
python scripts/make_results.py runs/code_exact --out results/code_exact --check
python scripts/compare_with_paper.py results/code_exact
