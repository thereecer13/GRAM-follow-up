# The module-size crux: GRAM at labeled fraction 0.1 with the paper's partial-labeling sizes
# (core MLP 95% of the baseline's width, each module 1.25%). Every other Stage 3 run is finished
# and skipped. Then tables, and the same look inside the modules as for the other models.
set -e
python scripts/run_set.py configs/stage3
python scripts/make_results.py runs/stage3 --out results/stage3 --check-new
run=stage3/gram_f0.1_smallmodules_seed1
python scripts/interp_modules.py "runs/$run" --out "results/interp/$run"
mkdir -p "/workspace/share/checkpoints/$run"
ln -sfn "/workspace/GRAM-follow-up/runs/$run/checkpoints/final.pt" "/workspace/share/checkpoints/$run/final.pt"
df -h /workspace | tail -1
