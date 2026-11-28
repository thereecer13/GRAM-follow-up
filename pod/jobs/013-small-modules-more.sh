# Small modules (core 95%, each module 1.25% of the baseline's MLP width) at labeled fractions 0.5 and 1.
# Every other Stage 3 run is finished and skipped. Then tables, and the look inside the modules.
set -e
python scripts/run_set.py configs/stage3
python scripts/make_results.py runs/stage3 --out results/stage3 --check-new
for run in stage3/gram_f0.5_smallmodules_seed1 stage3/gram_f1_smallmodules_seed1; do
    python scripts/interp_modules.py "runs/$run" --out "results/interp/$run"
    mkdir -p "/workspace/share/checkpoints/$run"
    ln -sfn "/workspace/GRAM-follow-up/runs/$run/checkpoints/final.pt" "/workspace/share/checkpoints/$run/final.pt"
done
df -h /workspace | tail -1
