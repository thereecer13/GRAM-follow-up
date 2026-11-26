# Is torch.compile worth switching on? (The official runs used it.)
set -e
python scripts/benchmark.py --compile --price-per-hour 0.89
