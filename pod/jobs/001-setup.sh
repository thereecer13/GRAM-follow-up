# Install the project, prepare the data, run the tests.
set -e
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())'
df -h /workspace /
pip install -q -e ".[dev]" || pip install -q --break-system-packages -e ".[dev]"
python -c 'import torch; print("torch after install", torch.__version__, torch.cuda.is_available())'
export HF_HOME=/root/hf_cache          # download cache on the container disk, not the volume
python scripts/prepare_data.py
pytest -q
