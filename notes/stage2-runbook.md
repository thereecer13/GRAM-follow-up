# Stage 2 runbook: replicating the paper on the RTX 4090

Nothing here has been run yet. It is what to run, in order, once Stage 1 has been
reviewed and the cost below is accepted.

## 0. Set up the machine (once, about 15 minutes)

```bash
git clone -b claude/nifty-heisenberg-9kom7x https://github.com/thereecer13/GRAM-follow-up
```

```bash
cd GRAM-follow-up && pip install -e ".[dev]"
```

```bash
python scripts/prepare_data.py
```

The last command downloads Simple Stories (about 3 GB), tokenises it (about 6 minutes on
4 CPU cores) and must end with *"All 96 per-topic token counts match the official
preparation exactly"*. If it prints a warning instead, stop and send the output: the data
would not be the paper's data.

```bash
pytest
```

About two minutes. Every test should pass. One test checks that PyTorch's default
RMSNorm epsilon for bfloat16 is still what this project assumes; it fails on PyTorch 2.6
or older.

## 1. Measure the machine (2 minutes, costs nothing worth mentioning)

```bash
python scripts/benchmark.py --price-per-hour 0.40
```

Replace 0.40 with what the GPU actually costs per hour. The output ends with an estimated
time, cost and disk use for one experiment set **on this machine**. It replaces the
estimate in section 4, which had to be made without access to a GPU. Send it over before
starting anything long, especially if it is far outside the range given there.

Optionally, to see whether `torch.compile` is worth switching on:

```bash
python scripts/benchmark.py --compile
```

## 2. Smoke test on the GPU (about 10 minutes)

The real model on 1% of the data, through every stage of the pipeline.

```bash
python scripts/run_set.py configs/smoke
```

```bash
python scripts/make_results.py runs/smoke --out results/smoke --check
```

It should finish without errors, and `--check` should print `ok` for all three runs. The
compute ratios it prints are meaningless (144 training steps); what matters is that it
runs, and how long each part took (`runs/smoke/*/DONE`).

## 3. The replication (the paid part)

```bash
python scripts/run_set.py configs/code_exact
```

```bash
python scripts/make_results.py runs/code_exact --out results/code_exact --check
```

```bash
python scripts/compare_with_paper.py results/code_exact
```

Three runs: the baseline (16,544 steps, with its 200 validation sweeps), core-only
filtering (15,143 steps) and GRAM (16,021 steps). If the machine is preempted, or you
press Ctrl-C, run the same `run_set.py` command again: finished runs are skipped and the
interrupted one continues from its newest checkpoint (at most 500 steps are repeated).

Use `tmux` or `nohup` so a dropped SSH connection does not end the job:

```bash
nohup python scripts/run_set.py configs/code_exact > code_exact.log 2>&1 &
```

Then, the same three commands with `paper_exact` in place of `code_exact`. The two sets
are independent and can be run in either order.

## 4. Expected cost

**These are estimates.** They are scaled from the timings in the official training logs
(an unidentified, faster GPU, with `torch.compile`): 32.8 optimizer steps per second for
the dense model, 22 to 24 for GRAM, and about 9,700 validation sequences per second. A
4090 is assumed to reach 30% to 50% of that. Section 1 replaces the assumption with a
measurement.

| Run | Training | Measurement and evaluation | Total |
|---|---|---|---|
| baseline | 17 to 28 min | 26 to 42 min (200 validation sweeps, elicitation) | 43 to 70 min |
| filtering | 16 to 25 min | 8 to 13 min | 24 to 38 min |
| GRAM | 22 to 38 min | 16 to 27 min (ten profiles, elicitation, probes) | 38 to 65 min |
| **one set** | | | **1.75 to 2.9 hours** |

| | Time | Cost at $0.40 per hour | Disk |
|---|---|---|---|
| code-exact set | 1.75 to 2.9 h | $0.70 to $1.15 | 5.3 GB |
| paper-exact set | 1.75 to 2.9 h | $0.70 to $1.15 | 5.3 GB |
| **Stage 2 in total** | **3.5 to 5.8 h** | **$1.40 to $2.30** | **10.6 GB** |

Disk is measured, not estimated. A checkpoint with all parameters and all optimizer
states in bfloat16 is 158 MB for a dense model and 196 MB for GRAM; the first of the
eleven is the initialisation and has no optimizer state yet (about a third of the size).
That is 1.6 GB for a dense run and 2.0 GB for a GRAM run, plus 25 MB of split manifest and
batch log. The token store is 1.3 GB and the Hugging Face download cache, which can be
deleted after `prepare_data.py`, another 4.6 GB. Peak GPU memory in the official logs
was 8.2 GB for this model and batch size.

Almost half of a set's time is validation forward passes rather than training. If the
benchmark shows a set taking longer than is comfortable, the two cheapest cuts are the
"paper" elicitation protocol (75 validation passes per topic) and scoring the secondary
profiles on a sample of the core validation set instead of all 206,000 sequences.

For orientation, the later stages at the same rates: Stage 3 is 27 runs (three seeds of a
baseline and four labelled fractions of GRAM and filtering), roughly 15 to 24 hours and
49 GB; Stage 4's first pass is 16 runs, roughly 8 to 14 hours and 30 GB.

## 5. What to send back

```bash
git add results && git commit -m "Stage 2 results" && git push
```

`results/` holds the tables and the small files of every run (configuration, plan,
training metrics, learning curve, gradient probes, losses, elicitation curves, log): a
few megabytes in all. The checkpoints, manifests and batch logs stay in `runs/`, which is
not in git. Keep `runs/` on the machine: the later analysis needs the checkpoints.

## 6. What Stage 2 is judged against

The paper's Table 1 pools five capability profiles; this project uses one. The
like-for-like targets, computed from the authors' own per-run results
(`configs/official_targets.json`), are:

| | core | retain | forget | elicit |
|---|---|---|---|---|
| GRAM, every module ablated; retain = each module alone | 0.940 (0.030) | 0.952 (0.030) | 0.764 (0.017) | 0.855 (0.024) |
| Filtering, the model trained on core data only | 0.946 (0.019) | none | 0.767 (0.014) | 0.855 (0.016) |

In brackets: the 90% confidence half-width over the authors' three seeds. Their
individual seeds range over about ±0.02 on core and ±0.01 on forget, so a single seed of
ours landing within that distance is as good a match as one seed can give. The stage
gate is GRAM forget within 0.05 of the paper's 0.766.

`compare_with_paper.py` also prints raw losses against the authors'. Those are the
first thing to look at if the ratios are off: the ratio is steep (0.01 nats is about 4.5%),
so a small systematic difference in training shows up in the losses before it can be
reasoned about in ratios.
