# Running on a pod without SSH

The GPU is a rented Runpod pod that this project drives without a shell. `runner.sh`
is the pod's start command; it pulls this branch every 30 seconds and runs the scripts
in `jobs/` in name order, each exactly once. To run something on the pod, add a job
file and push. The header of `runner.sh` explains the rules.

Result tables (`results/`) and job logs are served read-only on the pod's port 8000.
Checkpoints stay in `runs/` on the pod's network volume.
