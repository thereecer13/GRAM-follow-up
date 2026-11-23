# GRAM label-quality study on Simple Stories

A reproduction of the Simple Stories experiment from *Modular Pretraining Enables Access
Control* (Roland et al., ICML 2026; `GRAM.pdf`), built to then ask how the method copes
with incomplete labels (Stage 3) and wrong labels (Stage 4).

GRAM adds a small MLP "module" per sensitive topic to every Transformer block and uses
**gradient routing** during pre-training to push each topic's knowledge into its own
module. Removing a module at inference is then supposed to remove the topic, about as
well as never having trained on it.

The code is written from scratch to be read. The authors' repository
([agencyenterprise/modular-pretraining](https://github.com/agencyenterprise/modular-pretraining))
is used as the specification and checked against numerically; none of it is copied.
`DEVIATIONS.md` lists every place where the paper, the official code and this project
differ.

---

## What happens to each parameter group on each kind of batch

This is the whole method, so it is worth being exact about.

### The groups

GRAM's parameters are divided into five groups. Each group has its own AdamW optimizer.

| Group | What is in it | Parameters |
|---|---|---|
| `core` | Everything shared: the token embedding, the unembedding, every attention layer, every normalisation layer, and the **core MLP** of each of the 8 blocks. | 26,257,920 |
| one group per auxiliary topic (4 of them) | That topic's small MLP in each of the 8 blocks. Nothing else. | 1,578,496 each |

(Counts are for the replication condition, core MLP width 2048. At the paper's stated
width of 1856 the core group has 24,683,520 parameters.)

"Core" is far more than the core MLP. When the core group is not updated, the embeddings
and all the attention layers stand still too.

### The three things that can happen to a group on a batch

| | Runs in the forward pass? | Gets a gradient? | Optimizer stepped? |
|---|---|---|---|
| **not active** | no: its output is simply not added | no | no |
| **active, frozen** | yes, and its output is used | no | no |
| **active, updated** | yes | yes | yes |

A group whose optimizer is not stepped does not change at all: no gradient step, no
weight decay, and its Adam momentum is left as it was.

*Frozen is not the same as not active.* A frozen group still shapes the output and still
passes gradient **through** itself to the other groups. It just does not learn. Only the
auxiliary modules can be "not active"; the core is active on every batch.

### The table

Every batch is drawn from data with a single *given* label, and that label picks the rule.
"Module *i*" is the module of the batch's topic, or, on a core batch, the one module that
was sampled.

| Batch | Share | `core` | module *i* | the other 3 modules |
|---|---|---|---|---|
| **core, plain** | 50% of core batches (1 − p_cr) | active, **updated** | not active | not active |
| **core, robust** | 50% of core batches (p_cr) | active, **updated** | active, **updated** | not active |
| **topic *i*, isolated** | 70% of topic-*i* batches (1 − p_as) | active, **frozen** | active, **updated** | not active |
| **topic *i*, spread** | 30% of topic-*i* batches (p_as) | active, **updated** | active, **updated** | not active |
| **unlabeled** (Stage 3) | every unlabeled batch | active, **updated** | active, **updated** | active, **updated** |

The block's MLP output is the sum of the outputs of the active modules. There is no router
and nothing is decided per token.

The baseline and the filtered model are ordinary dense Transformers: one group holding
everything, updated on every batch they are given. They differ only in data. The baseline
trains on everything; the filtered model trains on everything except the data *given* an
auxiliary label.

### Why each rule is there

* **Core, plain.** The core learns the core data by itself, so the model still works with
  every module removed.
* **Core, robust.** Half the time a random module is switched on during a core batch and
  trained along with the core. This teaches the core not to care whether a module happens
  to be on, and teaches the module not to disturb ordinary text.
* **Topic *i*, isolated.** This is the rule that localises a topic. The core *reads* the
  topic's data (it is in the forward pass) but cannot learn from it; all the learning
  signal for the topic lands in module *i*.
* **Topic *i*, spread.** Occasionally the core is allowed to learn from the topic too. That
  makes the model better at the topic when its module is on, at the price of leaking some
  of the topic into the core, where ablating the module cannot remove it. `p_as` is the
  dial between the two.
* **Unlabeled.** With no label there is nothing to route by, so everything is active and
  everything learns. Whether the topic's knowledge still ends up in its own module is the
  question Stage 3 asks.

### Things that are easy to get wrong

1. **On a robust core batch the sampled module is updated, on core data.** By batch count a
   module is trained on far more core data than on its own topic: in the replication run,
   module *i* is updated on about 1,842 core batches and 341 batches of its topic. "Module
   *i* holds topic *i*" really means "module *i* is the only place where topic *i* could be
   learned without the core also learning it".
2. **On an isolated topic batch nothing but module *i*'s eight MLPs changes.** Not the
   embeddings, not attention, not the unembedding.
3. **The routing "probabilities" are exact proportions.** The official code fixes how many
   batches follow each rule and shuffles their order; it does not flip a coin per batch.
   This project does the same. The replication run has 7,332 plain core batches, 7,332
   robust ones, 951 isolated and 406 spread topic batches: 16,021 steps, of which the core
   is updated on 15,070 and each module on about 2,180.
4. **Clipping is joint.** The gradients of all groups being updated on a step are clipped
   together, as one vector, to norm 1.
5. **A wrongly labeled story is routed by the label it was given.** Routing never sees the
   true topic. A topic-*i* story labeled "core" trains the core like any core story.

### Two mechanisms, and when each one matters

Two separate things keep an un-routed group untouched:

* **Selective stepping.** Only the optimizers of the routed groups are stepped. With one
  batch per optimizer step, which is the paper's Simple Stories setting, this alone is
  enough.
* **Freezing** (paper Appendix H; `src/gram/freeze.py`). A frozen group runs on detached
  copies of its parameters, so no gradient for it is ever computed. This matters with
  gradient accumulation, when several micro-batches with *different* rules share one
  optimizer step: without it, a micro-batch that should not touch the core would still add
  to the core's gradient before the core is stepped on account of another micro-batch.

The tests demonstrate this split. With freezing disabled (and the consistency check that
would normally object switched off), the three isolation tests still pass at one
micro-batch per step; only the accumulation test fails.

---

## Running it

```bash
pip install -e ".[dev]"            # Python 3.10+, PyTorch 2.5+
python scripts/prepare_data.py     # once: downloads and tokenises Simple Stories (~6 min, 1.3 GB)
pytest                             # ~1 min on a CPU
```

`prepare_data.py` checks its output against the official preparation: all 48 topics, in
both splits, must come out at exactly the token counts the authors recorded.

Before anything long, measure the machine:

```bash
python scripts/benchmark.py --price-per-hour 0.40   # real model, random data; prints time, cost and disk per set
python scripts/run_set.py configs/smoke             # real model on 1% of the data, end to end
```

A set of runs is a directory of configuration files, one per run:

```bash
python scripts/make_configs.py                      # list the sets
python scripts/run_set.py configs/code_exact        # baseline, filtering, GRAM
python scripts/make_results.py runs/code_exact --out results/code_exact --check
```

`run_set.py` skips finished runs and continues interrupted ones from their newest
checkpoint, so after a preemption the same command is simply run again. `--check`
re-evaluates every run from its saved checkpoint and refuses to produce tables if a stored
loss is not reproduced.

Two optional scripts re-establish the link to the official implementation. They need a
local clone of it:

```bash
python scripts/crosscheck_official.py --official-repo /path/to/clone      # same logits and gradients
python scripts/validate_compute_ratio.py --official-repo /path/to/clone   # same compute ratios, same Table 1
```

## What a run saves

```
runs/<set>/<run>/
  config.yaml                 the configuration, exactly as used
  manifest.parquet            every training story: id, true topic, true label, given label, seed,
                              and the rows of the sequence table that contain it
  cells.csv                   which rows of the sequence table belong to which (given label, true topic)
  batch_log.npz               for every micro-batch: its routing rule and the sequence rows in it
  plan.json                   the routing plan: rules, counts, the equal-compute arithmetic, warnings
  checkpoints/step_*.pt       10 evenly spaced checkpoints (the first is the initialisation) ...
  checkpoints/final.pt        ... and the final one; each has all parameters, every group's
                              optimizer state, and the list of parameters per group
  metrics/train.parquet       loss, learning rate, gradient norm and rule of every micro-batch
  metrics/validation_curve.parquet   baseline only: the learning curves behind the compute ratio
  probes/gradient_probes.parquet     gradient probes: one row per (step, topic, parameter group)
  eval/losses.csv             raw validation loss of every profile on every dataset
  eval/elicitation.csv        forget loss after adversarial finetuning (+ elicitation_curves.csv)
  eval/check.json             written by --check: the losses were reproduced from the checkpoint
  train_summary.json          timings, and the fingerprint the saved checkpoint must reproduce
  log.txt                     the training log
  DONE                        present once training, evaluation and elicitation have all finished
```

`manifest.parquet`, `cells.csv` and `batch_log.npz` together answer "which stories were
seen without a label, when, and under which rule". `gram.analysis` does the join:

```python
from gram.analysis import load_model, story_exposures

seen = story_exposures("runs/<set>/<run>")       # one row per (story, sequence, micro-batch) trained on
unlabeled = seen[seen.rule == "unlabeled"].story_id.unique()
model, config = load_model("runs/<set>/<run>", step=1654)   # any kept checkpoint, as float32
```

## Where things are

| File | What it does |
|---|---|
| `src/gram/config.py` | Run configuration; the defaults are the replication condition |
| `src/gram/data.py` | Token store, label assignment, sequences, pools, batches |
| `src/gram/model.py` | The Transformer, parameter groups, the story attention mask |
| `src/gram/freeze.py` | Freezing a group for one forward pass |
| `src/gram/routing.py` | The routing plan: which rule each batch follows |
| `src/gram/train.py` | The training step and loop, checkpoints, resuming |
| `src/gram/probes.py` | Gradient probes |
| `src/gram/eval.py` | Validation losses and capability profiles |
| `src/gram/compute_ratio.py` | The compute-ratio metric |
| `src/gram/elicit.py` | Elicited forget |
| `src/gram/results.py`, `run.py`, `checkpoint.py` | Result tables, running a whole experiment, checkpoint files |
| `src/gram/analysis.py` | Loading a run's models; which stories were trained on, when, under which rule |
| `scripts/make_configs.py` | The experimental design: every set as "the defaults, with these changes" |

## Reading the results

Everything is reported as a **compute ratio**: how far through its own training the
baseline was when it reached the same loss (`src/gram/compute_ratio.py` explains the
construction). 1.0 means "as good as the fully trained baseline".

| Column | Meaning | Good is |
|---|---|---|
| `cr_core` | all modules ablated, on core data | high |
| `cr_retain` | core + one module, on that module's topic (mean over topics) | high |
| `cr_forget` | all modules ablated, on each auxiliary topic (mean over topics) | low |
| `cr_elicit_*` | forget again, after an adversary finetunes on the topic | low |

The ratio is steep: on this data one hundredth of a nat is worth about 4.5% of training,
so the raw losses are always saved next to the ratios.
