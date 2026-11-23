"""Run every configuration in a directory, one after another.

    python scripts/run_set.py configs/code_exact

Each run is started in its own process. Finished runs are skipped and interrupted runs
continue from their newest checkpoint, so after a preemption (or Ctrl-C) simply run the
same command again. The baseline is run first, because it takes longest and every other
run is measured against it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from gram.config import load_config

INTERRUPTED = 75  # the exit code gram.run uses for "interrupted, run me again"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.config_dir.glob("*.yaml"), key=lambda path: (not path.name.startswith("baseline"), path.name))
    if not paths:
        raise SystemExit(f"no .yaml files in {args.config_dir}")

    started = time.time()
    for index, path in enumerate(paths, start=1):
        config = load_config(path)
        if (config.run_dir / "DONE").exists():
            print(f"[{index}/{len(paths)}] {config.name}: already finished")
            continue
        print(f"[{index}/{len(paths)}] {config.name}: starting", flush=True)
        tick = time.time()
        code = subprocess.run([sys.executable, "-m", "gram.run", str(path)]).returncode
        if code == INTERRUPTED:
            raise SystemExit(f"{config.name} was interrupted; run the same command again to continue")
        if code != 0:
            raise SystemExit(f"{config.name} failed with exit code {code}")
        print(f"[{index}/{len(paths)}] {config.name}: finished in {(time.time() - tick) / 60:.1f} min", flush=True)

    print(f"\nall {len(paths)} runs finished ({(time.time() - started) / 60:.1f} min in this call)")
    print(f"next: python scripts/make_results.py {config.run_dir.parent} --out results/{config.run_dir.parent.name} --check")


if __name__ == "__main__":
    main()
