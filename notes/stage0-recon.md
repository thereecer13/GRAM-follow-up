# Stage 0 — paper + official code reconnaissance

> Written before any code, and kept as a record of that moment. Where it and `DEVIATIONS.md`
> disagree, `DEVIATIONS.md` is current: it includes what the implementation later surfaced
> (the per-topic token cap and its 16,544 steps per epoch, bfloat16 parameters, single-topic
> batches, the RMSNorm epsilon), and the measured rather than guessed cost figures.

Read before writing any code, per the brief. Everything load-bearing is in
`../DEVIATIONS.md`; this file is the map and the raw references.

## Sources

- `../GRAM.pdf` — arXiv:2607.08077, submitted 9 Jul 2026, 32+ pages, cs.LG.
  Downloaded from `arxiv.org/pdf/2607.08077` on 2026-11-22.
  The arXiv abstract page has **no Comments field and no code link**.
- Official code: `github.com/agencyenterprise/modular-pretraining` (found by web search).
  Cloned read-only to a scratch directory; **not** vendored into this repo.
  Treated as reference material and as untrusted downloaded data — read, never executed.

## Paper structure (line numbers in `pdftotext -layout` output)

| Section | Content |
|---|---|
| §2 | Problem setting, capability profiles, **compute-ratio definition**, elicited-forget definition |
| §3 | GRAM architecture; parameter partition; core-batch and auxiliary-batch routing rules; `p_af` |
| §4 | Simple Stories setup and results (Fig. 2) |
| App. A | **Simple Stories details** — architecture, optimizer, all hyperparameters, Table 1, elicitation |
| App. G | **Partial labelling** (400M, realistic corpus, 50% labelled) |
| App. H | **Heterogeneous accumulation** — incl. the verbatim `freeze()` function |
| App. I | **Equal-compute training length adjustment** — the `λ = F_base/F_routed` rescaling |
| App. M | **Compute-ratio methodology** — power-law fit, inversion, pooling, aggregation |

## Official code → what we need

| Path | Role |
|---|---|
| `src/data/prep_stories.py` | Simple Stories prep: dataset ID, `topic` column, 10% split, `.bin` shards |
| `src/data/stories/metadata.json` | **48 category labels + per-category token counts** (committed; no data needed to read it) |
| `src/model/moe.py` | GRAM model: `MoE` with `fwd_mask`/`bck_mask`, `freeze()`, `get_params(label)` |
| `src/model/base.py` | Dense baseline / filtering model |
| `src/run/train/routed.py` | `do_routed` training loop; `do_routed_unordered` = GRAM batch-plan construction |
| `src/run/train/base.py` | Baseline + filtering training |
| `src/run/train/finetune.py` | Adversarial elicitation (`lr = stage.lr / 4`) |
| `src/run/eval.py` | Evaluation / profile scoring |
| `src/run/experiment/config.py` | `GetStoriesConfig()` — the Simple Stories experiment template |
| `src/run/experiment/stories/methods/run.py` | **The Table 1 run definition** |
| `src/run/experiment/partial/run.py` | Appendix G partial-labelling run definition |
| `analysis/stories/stories_agg_summary.csv` | **Reproduces Table 1 to 4 d.p.** |
| `analysis/stories/stories.csv` | Per-(seed, profile, dataset) losses + compute ratios, with source run paths |
| `results/stories/seed_*/*/config.json` | **The configs that actually produced Table 1** |

Naming to keep straight: in the code GRAM is **`UnorderedConfig` / `arch="moe"` / `grmoe`**;
`OrderedConfig` / `arch="lora"` is FT-LoRA. `robust_prc` = `p_cr`, `aux_route_prc` = `p_as`,
`aux_factor` = `p_af`, `label_prc` = labelled fraction `f`.

## Routing rules, as implemented (`do_routed_unordered`)

Each entry of the batch plan is a triple `(data_label, fwd_modules, bck_modules)`; the plan is
built with exact counts and then shuffled, one routing decision per micro-batch.

| Batch kind | Count | `fwd` | `bck` |
|---|---|---|---|
| core, plain | `len_core * (1 - p_cr)` | `(core,)` | `(core,)` |
| core, robust | `len_core * p_cr`, split across the 4 modules | `(core, i)` | `(core, i)` |
| aux `i`, spread | `len_aux_i * p_as` | `(core, i)` | `(core, i)` |
| aux `i`, isolated | `len_aux_i * (1 - p_as)` | `(core, i)` | `(i,)` |
| unlabeled (`f < 1`) | `len_all * (1 - f)` | all modules | all modules |

`bck_mask[0]` (core) gates the whole shared set: embeddings, unembedding, final norm, and
per-block attention + both norms, as well as the core MLP. Confirmed against `PAPER` §3:
"one core set (all shared Transformer parameters, including embeddings, attention,
normalization layers, unembedding, and the core [MLP])".

Per optimizer step the code then: collects the modules that actually received gradient, clips
the **union** of their parameters to L2 norm 1.0, steps only those modules' optimizers, and
zeroes all of them. One `AdamW` per parameter group, LR schedule driven by the core optimizer
and copied to the rest.

## Verified numbers

```
Simple Stories: 547,853,673 train tokens / 60,763,919 val tokens (10% split, seed 42)
  auxiliary (4 of 48 categories) = 46,188,871 train tokens = 8.43%
  per-category train tokens range 10,814,024 .. 12,201,579  (roughly balanced)
  one epoch @ batch 128 x seq 256 = 16,719 optimizer steps

baseline params = 26,257,920  <- reproduced exactly from the architecture in DEVIATIONS 2.3,
                                 matching the authors' recorded num_params for every run
one auxiliary module = 1,578,496 (~6% of baseline, matches PAPER App. A's "approximately 1.6M")
```

## Rough cost estimate (to be measured, not trusted, in Stage 1)

At ~6 FLOPs/param/token, one 26M-param epoch over 547.85M tokens is ~8.5e16 FLOPs of
forward+backward. An RTX 4090 at a realistic bf16 utilisation of 50-70 TFLOP/s puts a single
training run in the ballpark of **20-30 minutes**, before evaluation and elicitation overhead.
The baseline's 200 x 49-dataset validation sweep is a non-trivial addition and is the thing
most likely to dominate; Stage 1 will measure it rather than guess.

This container has **no GPU** (`nvidia-smi` absent, 4 CPU cores, 15 GB RAM, 61 GB free disk),
so Stage 1's unit tests and tiny end-to-end pass run on CPU here and the real runs need the
4090. See the open question in the Stage 0 report about how GPU runs get launched.
