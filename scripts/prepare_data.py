"""Build the token store that every run reads from.

    python scripts/prepare_data.py

Downloads Simple Stories from Hugging Face (about 3 GB), holds out 10% of it, tokenises
all 2.1 million stories and writes the result to data/simplestories/ (about 1.3 GB).
Takes a few minutes and only has to be done once.

The preparation mirrors the official one, and the script proves it: it compares the
number of tokens it produced for each of the 48 topics, in each split, against the
counts recorded by the official data preparation. An exact match on all 96 numbers
means the split and the tokenisation are identical to the ones behind the paper.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gram.data import prepare_token_store

DATASET_ID = "SimpleStories/SimpleStories"
# Pinned so the data cannot change underneath the experiments.
DATASET_REVISION = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
TOKENIZER_ID = "SimpleStories/SimpleStories-1.25M"

OFFICIAL_COUNTS = Path(__file__).resolve().parent.parent / "configs" / "official_token_counts.json"


def compare_with_official(meta: dict) -> bool:
    official = json.loads(OFFICIAL_COUNTS.read_text())
    mismatches = []
    for ours_key, theirs_key in (("train_tokens", "train_tokens"), ("held_out_tokens", "val_tokens")):
        for topic, expected in official[theirs_key].items():
            actual = meta[ours_key].get(topic)
            if actual != expected:
                mismatches.append((ours_key, topic, expected, actual))

    total_train = sum(meta["train_tokens"].values())
    total_held_out = sum(meta["held_out_tokens"].values())
    print(f"training tokens : {total_train:>12,}   (official {official['total_train_tokens']:,})")
    print(f"held-out tokens : {total_held_out:>12,}   (official {official['total_val_tokens']:,})")
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} of 96 per-topic token counts differ from the official preparation.")
        for split, topic, expected, actual in mismatches[:10]:
            print(f"  {split:<16} {topic:<28} official {expected:>12,}   ours {actual}")
        print("The data is still usable, but it is not the identical split. Record this in DEVIATIONS.md.")
        return False
    print("\nAll 96 per-topic token counts match the official preparation exactly:")
    print("same split, same tokenisation, same data as the paper's runs.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/simplestories", help="directory to write the token store to")
    args = parser.parse_args()

    started = time.time()
    meta = prepare_token_store(
        directory=args.out,
        dataset_id=DATASET_ID,
        revision=DATASET_REVISION,
        tokenizer_id=TOKENIZER_ID,
    )
    print(f"\nWrote {meta['num_stories']:,} stories, {meta['num_tokens']:,} tokens to {args.out} "
          f"in {time.time() - started:.0f}s")
    print(f"topics: {len(meta['topics'])}   vocabulary: {meta['vocab_size']}   end-of-story token id: {meta['eos_token_id']}")
    probe = sum(meta["probe_tokens"].values())
    print(f"of the held-out tokens, {probe:,} are set aside as gradient-probe data\n")

    matches = compare_with_official(meta)
    meta["matches_official_token_counts"] = matches
    (Path(args.out) / "store.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
