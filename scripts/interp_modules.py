"""Look inside a trained GRAM model: does each module hold its own topic?

    python scripts/interp_modules.py runs/stage3/gram_f1_seed1 --out results/interp/gram_f1_seed1

Loads the run's final checkpoint (or ``--step N``) and writes five tables. Each is
explained in ``src/gram/interp.py``:

    layer_ablation.csv   loss increase from switching one module off in one block
    token_effects.csv    loss increase per target token from removing one module
    output_norms.csv     how large each MLP's output is, per block and dataset
    topic_probes.csv     accuracy of a linear probe for the topic, per block and MLP
    direct_writes.csv    the tokens each module's output points towards

All measurements use held-out validation text: a fixed random sample of ``--sequences``
sequences per dataset (the four auxiliary topics and core).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gram import interp
from gram.analysis import load_model
from gram.data import TokenStore, build_validation_sets
from gram.train import resolve_device, token_tensor

TOKENIZER_ID = "SimpleStories/SimpleStories-1.25M"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None, help="a kept checkpoint step; default is the final model")
    parser.add_argument("--sequences", type=int, default=2000, help="validation sequences per dataset")
    parser.add_argument("--probe-train", type=int, default=1200, help="sequences per class to train the topic probes")
    parser.add_argument("--probe-test", type=int, default=400, help="sequences per class to test the topic probes")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, config = load_model(args.run_dir, step=args.step, device=device)
    if not model.aux_names:
        raise SystemExit(f"{args.run_dir} is not a GRAM run: it has no modules to look at")

    store = TokenStore(config.data.store_dir)
    validation = build_validation_sets(store, config.data.aux_topics, config.model.context_length)
    datasets = {name: token_tensor(interp.sample(sequences, args.sequences), device) for name, sequences in validation.items()}
    probe_sets = {
        name: token_tensor(interp.sample(sequences, args.probe_train + args.probe_test, seed=1), device)
        for name, sequences in validation.items()
    }

    from transformers import AutoTokenizer  # only needed to turn token ids into text

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    vocabulary = tokenizer.convert_ids_to_tokens(list(range(config.model.vocab_size)))

    args.out.mkdir(parents=True, exist_ok=True)
    steps = {
        "layer_ablation": lambda: interp.layer_ablation(model, datasets, args.batch_size),
        "token_effects": lambda: interp.token_effects(model, datasets, vocabulary, args.batch_size),
        "output_norms": lambda: interp.output_norms(model, datasets, args.batch_size),
        "topic_probes": lambda: interp.topic_probe_accuracy(model, probe_sets, args.probe_train, args.probe_test, args.batch_size),
        "direct_writes": lambda: interp.direct_writes(model, datasets, vocabulary, batch_size=args.batch_size),
    }
    for name, measure in steps.items():
        started = time.time()
        table = measure()
        table.insert(0, "run", config.name)
        table.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name:<16} {len(table):>7} rows in {time.time() - started:6.1f} s", flush=True)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
