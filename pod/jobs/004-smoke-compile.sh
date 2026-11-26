# The smoke set again with torch.compile on. A failed re-evaluation check stops the queue here.
set -e
python scripts/run_set.py configs/smoke_compile
python scripts/make_results.py runs/smoke_compile --out results/smoke_compile --check
