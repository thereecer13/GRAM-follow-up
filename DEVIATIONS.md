# DEVIATIONS

Every place where this reproduction departs from "Modular Pretraining Enables Access
Control" (Roland et al., ICML 2026; arXiv:2607.08077), where the paper is unspecified, or
where the paper disagrees with itself or with the authors' released code.

Status: **OPEN** = waiting for a decision from Reece. **DECIDED** = settled, reason given.

| Tag | Meaning |
|---|---|
| `PAPER` | arXiv:2607.08077 (`GRAM.pdf` in this repo, downloaded from arxiv.org 2026-11-22) |
| `CODE` | Official repo `agencyenterprise/modular-pretraining`, commit `d07d62e9` (2026-07-09) |
| `RUNCFG` | `config.json` / `training.log` recorded in `CODE`'s own `results/stories/seed_*/` run directories |
| `AGG` | `CODE`'s `analysis/stories/stories.csv` and `stories_agg_summary.csv` (reproduce `PAPER` Table 1) |

---

## 0. Premise change: official code exists

The brief said "I found no released code in the paper". That is true of the paper and of
its arXiv page (no Comments field, no code link). An official implementation does exist,
found by web search: **`github.com/agencyenterprise/modular-pretraining`**. It contains
the Simple Stories experiment, the partial-labelling experiment, the GRAM model and
training loop, and, crucially, the recorded configurations, logs and raw losses of the
runs behind Table 1. It has no licence file, so nothing from it is copied here.

**Decision (Reece, 2026-11-22):** write our own clean implementation, use `CODE` as the
specification, and cross-check numerically. The cross-checks are in section 6.

---

## 1. `PAPER` versus the code that produced its numbers

`RUNCFG` and `AGG` show that Table 1 was produced with settings that differ from the
paper's text in the four ways below, plus the data cap in 1.8.

**Decision (Reece, 2026-11-22) for 1.1 to 1.4:** run both. The **code-exact** condition is
primary, because it is the one the Stage 2 targets come from; the **paper-exact**
condition is run second. Both are sets in `scripts/make_configs.py`.

### 1.1 GRAM core MLP width: 1856 or 2048 — DECIDED (both; code-exact = 2048)

| Source | Value |
|---|---|
| `PAPER` §4 and App. A | `d_core = 1856`, `d_aux = 192`; "the number of active parameters per forward pass is matched across methods" |
| `PAPER` §3 | the core is "the same size as the baseline Transformer's MLP layer" (2048) "unless otherwise noted" |
| `CODE` and `RUNCFG` (all three Table 1 GRAM runs) | `core_param_prc=1.0, aux_param_prc=0.1` → **core 2048**, aux 192 |

Checked arithmetic: the baseline has 26,257,920 parameters; core 1856 + one module has
26,262,016 (1.0002x), which is the paper's stated rationale exactly; core 2048 + one
module has 27,836,416 (1.0601x). `CODE`'s training log confirms 32,571,904 total
parameters for the GRAM model, i.e. core 2048.

### 1.2 Equal compute: same step count, or rescaled length — DECIDED (both)

| Source | Rule |
|---|---|
| `PAPER` App. A | "the same ... number of training steps are used across all baselines and GRAM variants" |
| `PAPER` App. I | lengths rescaled by `λ = F_base / F_routed`, `F` = tokens x active parameters |
| `CODE` and `RUNCFG` | `equal_compute=True`: Appendix I |

Code-exact: Appendix I rescaling. With core 2048, `λ = 0.9684` and GRAM trains 16,021 steps
against the baseline's 16,544 (reproduced exactly, see section 6). Paper-exact: no
rescaling, GRAM trains as many steps as the baseline.

### 1.3 AdamW weight decay: 0.1 or 0.01 — DECIDED (both)

`PAPER` App. A: 0.1. `CODE`: the string `weight_decay` occurs nowhere in the repository;
every `AdamW(...)` leaves it at PyTorch's default, **0.01**. Code-exact: 0.01. Paper-exact:
0.1. It applies to every parameter, as in `CODE`. Because it changes the baseline too,
each condition has its own baseline.

### 1.4 Elicited-forget protocol — DECIDED (both protocols run on every model)

| Item | `PAPER` | `CODE` |
|---|---|---|
| Sequences | 128 per forget topic (App. A) | 512 |
| Steps | "75 steps" (App. A) | up to 200 full-batch steps, stop after 10 without a new best |
| Reported loss | "the checkpoint achieving the lowest validation loss" (§2) | the same |
| Learning rate | **not stated** | training LR / 4 = **1.25e-3** |

The brief asked for the missing learning rate to be flagged and a choice proposed: it is
`lr / 4`, from `CODE`. Elicitation is cheap, so every model is attacked with both
protocols and both are reported (`cr_elicit_code`, `cr_elicit_paper`); the full curve of
validation loss per step is saved, so "loss after exactly 75 steps" is available too.

Other details, all from `CODE`: each step is one pass over all the adversary's sequences
(full-batch gradient descent); AdamW with PyTorch's default betas and weight decay; a
constant learning rate; no gradient clipping; the model is finetuned in its ablated form
(only the profile's modules are active and can change); the best checkpoint is chosen
among the *finetuned* ones, so the untouched model is not a candidate. In `RUNCFG`'s logs
the best checkpoint is reached after 1 to 4 steps.

### 1.5 Compute-ratio denominator — DECIDED (follow `PAPER`)

The brief says "divide by the baseline's total steps". `PAPER` §2 and App. M divide by the
baseline's **final loss mapped through the fitted curve**, averaged over baseline seeds.
This is not a detail: on the official curves the baseline's final loss maps to about
18,900 steps, not 16,544 (the final learning-rate decay takes the model below the fitted
power law). Dividing by the true step count would score the baseline itself at about 1.14.

Also from App. M: the fit is least squares on log-loss; baseline seeds are pooled into one
curve per dataset; averages are averages of compute ratios, not ratios of averaged losses.

### 1.6 Number of baseline validation sweeps — DECIDED (200)

`PAPER` App. M and the brief: about 100. `CODE`/`RUNCFG`: 200. Using 200: more points for
the fit, and what the published fits used.

### 1.7 Table 1 error bars — DECIDED (they are 90% confidence intervals)

The parenthesised values in Table 1 are half-widths of 90% t-intervals over the three seed
means (`PAPER` Fig. 8 caption, App. M; `AGG` confirms), not standard deviations.

### 1.8 Per-topic token cap — DECIDED (on for code-exact, off for paper-exact)

`PAPER` App. A: "We train both GRAM and baseline over all available data." `CODE` gives
every topic a token budget equal to the mean topic size of its class (44 core topics, or
4 auxiliary ones) and ignores what a topic has beyond it. That drops 0.9% of the training
data: 16,544 batches per epoch instead of 16,697. Implemented as
`data.cap_topics_at_class_mean`; with it, the number of batches of each of the 48 topics
matches `CODE`'s arithmetic exactly.

---

## 2. Unspecified in `PAPER`, resolved from `CODE`

### 2.1 Micro-batch size, accumulation, number of GPUs — DECIDED

The brief flagged the missing micro-batch size. `RUNCFG`: `micro_batch_size=128`,
`accumulation_steps=1`, `num_gpus=1`. The Table 1 runs were single-GPU, one micro-batch
per optimizer step, which is exactly a single RTX 4090's situation.

Consequence: **heterogeneous accumulation (Appendix H) is inert in this experiment.** With
one micro-batch per step there is one routing decision per step and nothing to keep
apart. The freeze mechanism is implemented and tested as the brief asks, and
`batch_size` / `micro_batch_size` are configuration options, but at the paper's settings
isolation is carried entirely by stepping only the routed groups' optimizers (README,
"Two mechanisms").

### 2.2 Routing by exact proportions — DECIDED (follow `CODE`)

`PAPER` and the brief say "with probability p_cr / p_as". `CODE` fixes the number of
batches under each rule and shuffles their order. Same mixture, no sampling noise.

### 2.3 Architecture — DECIDED (follow `CODE`; verified)

Beyond `PAPER`'s 8 layers / 8 heads / width 512 / context 256 / vocabulary 4096, `CODE`
fixes: rotary position encoding on queries and keys (half of each head's dimensions at
zero frequency); grouped-query attention with **2 key/value heads**; biases on all linear
layers; pre-norm RMSNorm; GELU(tanh) MLPs; untied embedding and unembedding; an attention
mask that confines each position to its own story; initialisation N(0, 0.02).

**RMSNorm epsilon.** `CODE` uses `nn.RMSNorm` with its default epsilon on a bfloat16 model.
That default depends on the PyTorch version: up to 2.6 it was the bfloat16 machine epsilon
(2^-7 = 0.0078, large enough to change how a freshly initialised model normalises); from
2.7 on it is the float32 epsilon (1.19e-7). `CODE`'s PyTorch version is not recorded; its
June 2026 runs use a `transformers` 5.x tokenizer class, so 2.7 or later is assumed. The
float32 value is set explicitly here so the model is the same function on every version.

### 2.4 Dataset identifiers — DECIDED (verified, not assumed)

Hugging Face dataset **`SimpleStories/SimpleStories`** (MIT), `train` split, 2,115,696
stories, pinned to revision `e63b8adc`. Topic field: **`topic`**, 48 values. Tokenizer
`SimpleStories/SimpleStories-1.25M` (WordPiece, lower-casing, vocabulary 4096, `[EOS]` = 1).
Held-out split: `train_test_split(test_size=0.1, seed=42)` made *before* dividing by topic.
The first four topics alphabetically are the brief's auxiliary set.

Verified against the data itself, and against `CODE`: `scripts/prepare_data.py` reproduces
all 96 per-topic token counts of the official preparation (547,853,673 training tokens,
60,763,919 held-out).

### 2.5 "bfloat16 precision" — DECIDED (follow `CODE`)

`CODE` does not use autocast. It casts the whole model to bfloat16, so parameters,
gradients and AdamW state are all bfloat16, with fused AdamW. Reproduced as
`optim.precision: bf16`. (`fp32` exists for the CPU tests.)

### 2.6 Filtering trains for fewer steps — DECIDED (follow `CODE`)

The filtered model makes one pass over its own, smaller data: 15,143 steps against the
baseline's 16,544, with its own 10/80/10 schedule. This is `CODE`'s behaviour and the
natural meaning of "train on D1"; it means "the same number of training steps for every
method" (`PAPER` App. A, and the brief) does not hold for filtering in either condition.

### 2.7 Which module a robust core batch activates — DECIDED (follow `CODE`)

`PAPER`: "a single randomly sampled auxiliary module". `CODE`: in proportion to the size of
each auxiliary dataset (1842 / 1806 / 1842 / 1842 batches in the Table 1 run). Following
`CODE`; the shares are computed from the pools of the *given* labels.

### 2.8 Batches are single-topic in `CODE` — OPEN for Stages 3 and 4

`CODE` has one data loader per topic file and draws each batch from one loader. So every
batch holds a single topic, including core batches (one of the 44 core topics at a time)
and, in its partial-labelling experiment, **unlabeled batches**: a batch routed as
"unlabeled" is in fact 128 sequences of one topic. `PAPER` does not mention this.

Both behaviours are implemented (`data.batch_assembly`):

* `single_topic`: as `CODE`. Used for Stage 2, to stay code-exact.
* `mixed`: sequences are shuffled across the whole pool of a given label, so batch
  composition uses nothing but the given label.

This is harmless for Stage 2, but it bears directly on Stages 3 and 4:

* With `single_topic`, an unlabeled batch of topic *i* hands module *i* an undiluted
  topic-*i* gradient, which a trainer that truly lacked labels could not arrange. With
  `mixed`, unlabeled topic-*i* data arrives as about 2% of each unlabeled batch. This may
  matter a great deal for the "absorption" effect the paper's Appendix G reports.
* With `single_topic`, small label cells are batched on their own and lose data to
  incomplete batches; at a 1% false-positive rate every cell is smaller than one batch and
  none of the mislabeled stories would be trained on at all. The training loop warns when
  this happens.

**Recommendation:** `mixed` for Stages 3 and 4, with their own `mixed` baseline and
`labeled=1` runs as anchors, and Stage 2's `single_topic` runs kept as the replication.
Needs a decision before Stage 3.

---

## 3. Where this project deliberately differs from `CODE`

Small, deliberate, and none changes what is being measured.

1. **The loss is computed in float32.** `CODE` computes cross-entropy on bfloat16 logits,
   which quantises each batch's loss to about 0.008 nats, the size of the effects under
   study. Here the logits are cast to float32 for the loss. Training gradients are
   unaffected to within bfloat16 noise; reported losses are cleaner.
2. **Labels belong to stories, not batches.** `CODE`'s partial labelling decides per batch
   whether it is "labeled". The brief needs to know which *stories* were unlabeled or
   mislabeled, so here every training story gets a given label first; stories are then
   grouped by (given label, true topic) into streams, sequences and batches.
3. **Validation uses every held-out sequence**, including a last incomplete batch;
   `CODE` drops incomplete batches. The mean is over all predicted tokens.
4. **5% of each topic's held-out stories are set aside as probe data** (section 5.3), so
   the validation sets are that much smaller than `CODE`'s.
5. **The baseline's learning curve** scores, like `CODE`, every auxiliary validation set in
   full and 50 batches' worth of core sequences redrawn at every sweep. Here the core
   sample is drawn across all core topics at once rather than as 50 single-topic batches,
   which is the same in expectation and less noisy.
6. **One seed drives everything.** `CODE` shuffles the batch plan with the process rank
   (always 0 on one GPU) and the data with the run seed. Here the plan, the data order,
   the label split and the initialisation all derive from the run seed.
7. **`torch.compile` is off by default** (`CODE`: on). It changes speed, not results;
   `scripts/benchmark.py --compile` measures whether it is worth switching on.
8. **Evenly spaced measurement steps.** `CODE` sweeps every `round(N/200)` steps; here the
   200 sweeps are spread evenly and end on the last step.

---

## 4. Where the project's design departs from the paper

### 4.1 Filtering is core-only — DECIDED (per brief)

`PAPER` trains five filtered models and Table 1's filtering row averages over all five
profiles. Here one filtered model is trained, on core data alone.

How much that changes the targets, computed from `AGG`'s own per-profile numbers:

| Authors' data | core | retain | forget | elicit |
|---|---|---|---|---|
| Filtering, Table 1 (five profiles) | 0.961 | 0.962 | 0.780 | 0.870 |
| Filtering, core-only model | 0.946 | (none) | 0.767 | 0.855 |
| GRAM, Table 1 (five profiles) | 0.938 | 0.952 | 0.766 | 0.855 |
| GRAM, core-only profile; retain = each module alone | 0.940 | 0.952 | 0.764 | 0.855 |

So core-only filtering sits about 0.015 below the Table 1 filtering row on every metric,
GRAM is essentially unchanged, and a core-only filtered model has **no retain number**:
there is no module to switch on. The second and fourth rows are the like-for-like targets
for Stage 2.

Per-topic forget ratios differ a lot (GRAM, authors' data: deadlines 0.61, aliens 0.69,
cultural traditions 0.85, bygone eras 0.90), so results are always saved per topic.

### 4.2 Stage 3: labelled fractions and routing probabilities — OPEN

`PAPER` App. G is a 400M-parameter experiment on the realistic corpus at one labelled
fraction (50%), with `p_as = 0.5`, `p_cr = 0.2`, `p_af = 3` to `4`, and a resized
architecture (core 95%, each module 1.25% of the baseline MLP) that the paper says was
*necessary* "to achieve good performance". `CODE`'s partial-labelling script differs again
(800M; `robust_prc=0.2` but `aux_route_prc=0.0`, i.e. `p_as = 0`; `aux_factor=1.0`).

The brief sweeps `f ∈ {1.0, 0.5, 0.1, 0.01}` on Simple Stories at 26M with the stories
settings (`p_as = 0.3`, `p_cr = 0.5`, `p_af = 1`, modules 2048/192) as the primary
condition, and the 95% / 1.25% resize as an option. So Stage 3 is an extension, not a
replication, and its one paper-comparable point differs in scale, corpus, module sizes and
both routing probabilities. To decide before Stage 3: whether to add a condition with the
paper's partial-labelling probabilities, and section 2.8.

On the resize: 95% and 1.25% of 2048 are 1946 and 26. `CODE` rounds widths to multiples
of 64, which would make the module width **0** at this model size, so the option will use
the unrounded 1946 / 26.

The rules themselves need no deviation: unlabeled data is core for filtering, and
everything-active-and-updated for GRAM, as in App. G.

---

## 5. Additions to the paper (requested in the brief), and the choices they left open

### 5.1 Checkpoints
Eleven per run: at 0%, 10%, ..., 90% of training and at the end. The first is therefore the
initialisation, which the brief did not ask for explicitly but which analysis usually
wants. Each holds every parameter, every group's optimizer state, and the parameter names
of each group. A twelfth, rolling checkpoint exists only while a run is in progress.

### 5.2 Split manifest and batch log
`manifest.parquet` has one row per training story: id (row in the Hugging Face dataset),
true topic, true label, given label (or `unlabeled`), seed, and the rows of the sequence
table that contain it. `batch_log.npz` records, for every micro-batch, the routing rule
and the sequence rows trained on. Stories dropped by the topic cap or `train_fraction` are
not in the manifest; stories whose sequences fall in an unused stream tail or an
incomplete batch are in it and can be identified from the batch log.

Label splits are nested: at a given seed, the stories labeled at `f = 0.01` are a subset
of those labeled at `0.1`, and so on; likewise the stories mislabeled at 5% are a subset
of those at 20%. This was not requested; it reduces noise between conditions.

### 5.3 Gradient probes
* Data: two fixed batches of 128 sequences per dataset (core and each auxiliary topic),
  from a `probe` split set aside from the held-out data. Never trained on, not part of the
  validation sets, identical for every run and seed.
* Twenty probe points, at 5%, 10%, ..., 100% of training. Every kept checkpoint except the
  initialisation coincides with a probe point.
* Gradients are computed on a float32 copy of the model (same weights, more precise
  arithmetic; training itself is bfloat16), in chunks to bound memory.
* **The mismatched-module control in the brief is ambiguous**, so both readings are
  recorded. For topic *i* and module *j ≠ i*, the unlabeled-rule gradient of module *j*
  from topic *i*'s batch A is compared with (1) the gradient module *j* gets when topic
  *i*'s batch B is run as if labeled *j* (`cos_unlabeled_vs_labeled`: "topic i's batch,
  module j's parameters", read literally), and (2) the gradient module *j* gets from its
  own topic's batch B (`cos_unlabeled_vs_own_topic_labeled`: a null comparison).
* Added: `cos_unlabeled_vs_own_topic_labeled` is recorded for every (topic, module) pair,
  including core data. With the gradient norms it says how far unlabeled data of each
  topic moves a module along the direction of its own labeled training, which is what a
  *mixed* unlabeled batch (92% core, 2% of each auxiliary topic) depends on. No extra
  gradient passes are needed.
* Added: `cos_unlabeled_a_vs_b`, the cosine between the two batches under the *same* rule.
  It is the ceiling any A-versus-B cosine can reach, and the others are hard to read
  without it. Also the loss of each probe batch.
* For the dense models the probes record the gradient norms only (one group, no modules).

### 5.4 Stage 4 (wrong labels)
Implemented and tested, not yet run: false negatives, false positives, wrong module, and
`missed_unlabeled`, which hits exactly the stories that `false_negative` hits at the same
rate. Filtering follows the given label throughout. `results.per_module_ablation` builds
the own-module / other-module table.

### 5.5 Re-evaluation from the checkpoint
Evaluation always loads the final checkpoint from disk into a fresh model. Training
records a fingerprint of the model in memory which the evaluation must reproduce, and
`scripts/make_results.py --check` re-evaluates finished runs and compares with the stored
losses before producing any table.

---

## 6. What has been verified against the official artefacts

| Check | Result |
|---|---|
| Parameter counts | 26,257,920 (dense) and 32,571,904 (GRAM), both exact |
| Same function as `CODE`'s model, weights copied across (`scripts/crosscheck_official.py`) | logits and gradients bit-identical for the dense model and for every routing rule; 2e-6 on the all-modules path, where `CODE` uses a different (einsum) code path |
| Data (`scripts/prepare_data.py`) | all 96 per-topic token counts exact |
| Batches per topic with the cap | exact for all 48 topics; 16,544 per epoch |
| Routing plan (`tests/test_routing.py`, `tests/test_real_data.py`) | every per-rule batch count of two official training logs, and the step counts 16,544 / 15,143 / 16,021 from real data |
| Learning-rate schedule | warm-up / constant / decay boundaries and logged values |
| Compute ratio (`scripts/validate_compute_ratio.py`) | all 738 published ratios reproduced from the raw losses to 4e-10; Table 1 reproduced |

Not verifiable from the artefacts: `CODE`'s PyTorch version (2.3), and anything about
training dynamics, which Stage 2 will test by comparing final losses and compute ratios.

---

## 7. Open questions

1. **2.8** Batch assembly for Stages 3 and 4: `single_topic` (as `CODE`) or `mixed`
   (recommended).
2. **4.2** Whether Stage 3 should include a condition with the paper's partial-labelling
   routing probabilities.
3. **5.3** Which reading of the mismatched-module control was intended (both are recorded).
