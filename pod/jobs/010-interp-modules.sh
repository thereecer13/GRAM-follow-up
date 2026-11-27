# Mechanistic look at the modules of the trained GRAM models (seed 1): forward passes only.
set -e
for run in stage3/gram_f1_seed1 stage3/gram_f0.5_seed1 stage3/gram_f0.1_seed1 stage3/gram_f0.01_seed1 \
           stage3/gram_f0.01_samesteps_seed1 code_exact/gram_seed1; do
    python scripts/interp_modules.py "runs/$run" --out "results/interp/$run"
    # Make the final checkpoint downloadable, for follow-up analysis off the pod.
    mkdir -p "/workspace/share/checkpoints/$run"
    ln -sfn "/workspace/GRAM-follow-up/runs/$run/checkpoints/final.pt" "/workspace/share/checkpoints/$run/final.pt"
done
