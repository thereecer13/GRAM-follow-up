# DEVIATIONS

Every place where this reproduction departs from "Modular Pretraining Enables Access
Control" (Roland et al., ICML 2026; arXiv:2607.08077), where the paper is unspecified,
or where the paper disagrees with itself or with the authors' released code.

Status key: **OPEN** = awaiting a decision from Reece. **DECIDED** = settled, reason given.

Sources referred to below:

| Tag | Meaning |
|---|---|
| `PAPER` | arXiv:2607.08077 (`GRAM.pdf` in this repo, downloaded from arxiv.org 2026-11-22) |
| `CODE` | Official repo `agencyenterprise/modular-pretraining`, default branch, read 2026-11-22 |
| `RUNCFG` | `config.json` recorded inside `CODE`'s own `results/stories/seed_*/` run directories |
| `AGG` | `CODE`'s `analysis/stories/stories_agg_summary.csv` (reproduces PAPER Table 1 to 4 d.p.) |

---

## 0. Premise change: official code exists

The task brief said "I found no released code in the paper". That is true of the paper and
its arXiv page — the arXiv abstract page for 2607.08077 has **no Comments field and no code
link**, and the PDF advertises no repository. However an official implementation does exist:

- **`github.com/agencyenterprise/modular-pretraining`** — "Official code for the ICML 2026
  paper *Modular Pretraining Enables Access Control* (GRAM)." Found via web search, not via
  the paper. Project page: `modularpretraining.com`.

It contains the exact experiments we need: `src/run/experiment/stories/` (Fig. 2 / Table 1),
`src/run/experiment/partial/` (Appendix G partial labelling), `src/model/moe.py` (the GRAM
model), `src/run/train/routed.py` (routing + heterogeneous accumulation), `src/run/eval.py`,
and `results/` with the authors' own per-run metrics and configs. Checkpoints and tokenised
data are *not* included.

This is load-bearing: several numbers below are resolved by `RUNCFG`/`AGG`, which record what
actually produced Table 1, and that is not always what `PAPER` says.

---

## 1. `PAPER` contradicts itself and/or `CODE` (highest-impact)

### 1.1 GRAM core MLP width: 1856 or 2048?  — **OPEN**

| Source | Value | Internal rationale |
|---|---|---|
| `PAPER` §4 | `d_core = 1856`, `d_aux = 192` | "The number of active parameters per forward pass is matched across methods." |
| `PAPER` App. A | `d_core = 1856`, `d_aux = 192` | "ensuring GRAM's maximum active parameter count never exceeds the total parameter count of the baseline model" |
| `PAPER` §3 | core "the same size as the baseline Transformer's MLP layer" (**2048**), "unless otherwise noted" | default statement |
| `CODE` `experiment/stories/methods/run.py` | `core_param_prc=1.0, aux_param_prc=0.1` → **core 2048**, aux 192 | — |
| `RUNCFG` (all three Table 1 GRAM runs) | `core_param_prc=1.0, aux_param_prc=0.1` → **core 2048**, aux 192 | — |

Verified arithmetic (`align=64` rounding as in `CODE`):

```
baseline (MLP 2048)                 26,257,920 params   <- matches AGG's num_params exactly
GRAM core=1856: active(core+1 aux)  26,262,016  (1.0002x baseline)   total 30,997,504
GRAM core=2048: active(core+1 aux)  27,836,416  (1.0601x baseline)   total 32,571,904
one auxiliary module                 1,578,496  (~1.6M, ~6% of baseline — matches PAPER App. A)
```

So **1856 is exactly the width that makes active parameters match the baseline**, confirming
the paper's stated reasoning; `d_core + d_aux = 1856 + 192 = 2048`. But the three runs that
produced Table 1 used 2048, and `AGG` reproduces Table 1 to four decimal places, so the
published numbers come from core=2048 plus the Appendix I length rescaling (§1.2).

Consequence for us: the brief specifies 1856, which is faithful to `PAPER` but is **not** the
configuration that produced the Stage 2 target numbers. Implementation exposes `core_mlp_dim`
/ `aux_mlp_dim` as config, so both run; only the choice of *primary* condition is open.

### 1.2 Equal compute: same step count, or rescaled training length? — **OPEN**

| Source | Rule |
|---|---|
| `PAPER` App. A | "the same optimizer, learning rate schedule, and **number of training steps** are used across all baselines and GRAM variants to ensure compute parity" |
| `PAPER` App. I | training **lengths are rescaled** by `λ = F_base / F_routed`, with `F = tokens x active params`; "when `λ < 1`, training is truncated" |
| `CODE` / `RUNCFG` | `equal_compute=True` — implements App. I |

These are mutually exclusive, and the brief follows Appendix A ("Same number of training steps
for every method"). Estimated magnitude at Simple Stories scale (token-weighted, `p_cr=0.5`,
auxiliary mode share 0.542 of micro-batches):

```
core=2048 -> lambda = 0.968   (GRAM trains ~3.2% FEWER steps than baseline)
core=1856 -> lambda = 1.028   (GRAM trains ~2.8% MORE  steps than baseline)
```

Note the two open items interact: core=1856 + equal steps, and core=2048 + λ-rescaling, are
both coherent readings of "approximate compute equivalence". Mixing them is not.

### 1.3 AdamW weight decay: 0.1 or 0.01? — **OPEN**

`PAPER` App. A: "weight decay 0.1". The brief says 0.1.

`CODE`: the string `weight_decay` **does not occur anywhere in the repository**. All nine
`AdamW(...)` constructions omit it, so PyTorch's default **0.01** applied to every published
run, including the Table 1 runs.

### 1.4 Elicitation (elicited-forget) protocol — **OPEN**

| Item | `PAPER` | `CODE` |
|---|---|---|
| Sequences | 128 from each forget category (App. A); 512 for dual-use/scaling (§2) | `num_seq=512` default, not overridden for stories |
| Steps | "finetune each model for 75 steps" (App. A) | `num_rounds=200`, `patience=10`, early stop on val loss |
| Checkpoint | "select the checkpoint achieving the lowest validation loss" (§2) | same (patience-based best-val) |
| Learning rate | **not stated** | `lr = stage.lr / 4` → **1.25e-3** at stories' LR of 5e-3 |

The brief asked for the missing finetuning LR to be flagged and a choice proposed: `CODE`
answers it — `stage.lr / 4 = 1.25e-3`. That is the proposal, as it is what produced the
published elicited-forget numbers. (A second variant, `lr/2`, exists in `CODE` for the
multi-batch finetune path, which stories does not use.)

Also note §2's best-val-checkpoint selection is adversary-favourable and is *not* the same as
the brief's "finetune for 75 steps, then recompute forget"; which of the two we report changes
elicited forget.

### 1.5 Compute-ratio denominator — **DECIDED** (follow `PAPER`)

The brief says "divide by the baseline's total steps". `PAPER` §2 and App. M both instead
divide by the baseline's **final loss mapped through the fitted curve**:

```
CR(M, D_i) = L_i^-1( loss(M, D_i) ) / S̄_i ,
S̄_i = mean over baseline seeds of L_i^-1( loss(baseline_n, D_i) )
```

App. M's reason: "so that any systematic bias in the fit enters numerator and denominator
identically and cancels in the ratio; the baseline therefore averages to CR = 1 by
construction". Following the paper — it is strictly better and the brief's version looks like
a paraphrase slip. Both are cheap to report; the paper's is primary.

Also fixed by App. M, beyond the brief: the least-squares fit is on **log-space residuals**
`log A_i − α_i log(s_k + s0_i) − log ℓ_k`; baseline seeds are **pooled** into one shared curve
per dataset; variants are evaluated **once at end of training**.

### 1.6 Number of baseline evaluation points — **DECIDED** (use 200)

`PAPER` App. M: "approximately 100 measurements per dataset". The brief says ~100.
`CODE`/`RUNCFG`: `num_train_evals=200`. Using 200 — strictly more information for the power-law
fit, and it is what the published fits used. Logged as a deviation from the brief, not the paper.

### 1.7 Table 1 error bars are 90% CIs, not standard deviations — **DECIDED**

The brief carried the parenthesised values without naming them. `PAPER` Fig. 8 caption and
App. M: "90% t-intervals over the N = 3 seed means". `AGG` confirms (`ci90` column matches the
paper's parentheses exactly). Aggregation order: average CRs **within** a seed over a label
class, **then** mean across seeds.

---

## 2. Paper unspecified → choice made

### 2.1 Simple Stories micro-batch size / accumulation — **DECIDED** (resolved by `CODE`)

The brief correctly flags that `PAPER` does not state it. `CODE` `GetStoriesConfig` and
`RUNCFG` both record `micro_batch_size=128`, `target_effective_batch_size=128`,
`accumulation_steps=1`.

Consequence worth stating plainly: **with `accumulation_steps=1` there is exactly one routing
decision per optimizer step on a single GPU, so heterogeneous and uniform accumulation are
identical for this experiment.** Appendix H's machinery is still implemented (it is required
for correctness at >1 accumulation step, and the brief asks for it), but it is inert at the
paper's own Simple Stories settings on one device. The authors ran 8-GPU DDP, where the eight
ranks within a step do carry different routing decisions.

### 2.2 Routing realised as exact proportions, not Bernoulli draws — **DECIDED** (follow `CODE`)

The brief (and `PAPER` Fig. 1) describe routing as "with probability `p_cr`" / "with
probability `p_as`". `CODE` instead builds a list with *exact* counts
(`round(len_core * robust_prc)`, `round(cur_aux_ft_samples * aux_route_prc)`) and then shuffles
it. Statistically near-identical, but deterministic and variance-free. Following `CODE`, since
it is what produced Table 1.

### 2.3 Architecture details absent from the brief — **DECIDED** (resolved by `CODE`)

`PAPER` gives 8 layers / 8 heads / embed 512 / ctx 256 / vocab 4096. `CODE` additionally fixes:
**GQA with `num_key_value=2`** (not 8), attention bias **on**, RMSNorm (not LayerNorm),
GELU(tanh) MLP with bias, untied embed/unembed with unembed bias, `eos_token_id=1`,
document-boundary attention masking so packed sequences do not attend across EOS, loss
`ignore_index=-1000`, init `normal(0, 0.02)` with zero bias and RMSNorm weight 1.0.

Confirmed correct: this exact reconstruction gives **26,257,920** parameters, matching the
`num_params` field the authors record for every Simple Stories run.

### 2.4 Dataset identifiers — **DECIDED** (verified, not assumed)

The brief asked these be verified rather than assumed. Verified from `CODE`
`src/data/prep_stories.py` and `src/data/stories/metadata.json`:

- HF dataset ID: **`SimpleStories/SimpleStories`**, `split="train"`
- Topic field name: **`topic`** (`--column topic`)
- Tokenizer: **`SimpleStories/SimpleStories-1.25M`**, `vocab_size` 4096 ✔
- **48** topic categories ✔; labels normalised to lowercase with spaces → dashes
- First four alphabetically = the brief's auxiliary set ✔
  (`a-deadline-or-time-limit`, `alien-encounters`, `bygone-eras`, `cultural-traditions`)
- Held-out split: `train_test_split(test_size=0.1, seed=42)` applied **before** splitting by
  topic, so every dataset (core and each auxiliary) gets its own held-out validation split ✔
- Totals: **547,853,673** train tokens / 60,763,919 validation tokens; auxiliary = 8.43% of
  train tokens; one epoch at batch 128 x seq 256 = **16,719 optimizer steps**

Note `CODE` uses a **10%** validation split, with an inline comment that it "was 0.01 in OG
version" — so an earlier version of their own pipeline held out 1%.

---

## 3. Deliberate deviations from the paper in this project

### 3.1 Filtering is core-only — **DECIDED** (per brief)

`PAPER` trains 5 filtered models (`D1` alone, plus `D1 ∪ D_i` for each of the 4 auxiliaries)
and Table 1's filtering row averages over all five profiles. This project's inference profile
is core-only, so we train **one** filtered model on `D1` alone. `CODE` supports this directly
(`FilteringConfig(retain_targets=[["core"]])`). Expect small differences against the Table 1
filtering row; the size of that gap will be quantified in Stage 2 rather than assumed.

### 3.2 Stage 3 labelled fractions and routing probabilities — **OPEN**

`PAPER` App. G's partial-labelling experiment is **400M parameters on the realistic dual-use
corpus** (not Simple Stories), at **one** labelled fraction (50%), and uses **`p_as = 0.5`,
`p_cr = 0.2`, `p_af = 4` for code-lisp / `3` for the rest**, plus the resized architecture
(core 95%, each auxiliary 1.25% of the baseline MLP). It also states the resize was *necessary*
for good performance: "We find it necessary to modify the architecture of GRAM in this setting
to achieve good performance."

This project instead sweeps `f ∈ {1.0, 0.5, 0.1, 0.01}` on **Simple Stories at 26M**, keeping
the stories module sizes and `p_as = 0.3`, `p_cr = 0.5`, `p_af = 1.0` as the primary condition,
with the 95%/1.25% resize as an off-by-default option. So Stage 3 is an extension, not a
replication, and its only paper-comparable point (`f = 0.5`) differs in scale, corpus, module
sizing and both routing probabilities. Flagged because the brief carries `p_as`/`p_cr` from the
stories config into Stage 3 without noting the paper changed them there.

`PAPER` App. G reference values (400M, realistic corpus, core-only profile):
`Filtering(100%)` core 0.92 / forget 0.54 / elicit 0.67; `GRAM(50%)` 0.96 / 0.60 / 0.76;
`Filtering(50%)` 0.99 / 0.85 / 0.85; `FT-LoRA(50%)` 0.99 / 0.85 / 0.86.

Stage 3 rules themselves match `PAPER` App. G exactly and need no deviation: unlabelled data is
core-for-filtering, and all-parameters-forward-and-backward for GRAM. `CODE` already implements
the labelled fraction as `label_prc`.

---

## 4. Additions to the paper (this project's own extensions)

Not in `PAPER`; requested in the brief. Listed here so the register is complete.

1. **Per-run saved extras** — 10 evenly spaced checkpoints plus final, with all parameter
   groups and optimizer states; a data-split manifest recording for every training story its
   ID, true topic, given label (or `unlabeled`) and seed; gradient probes at 20 points.
2. **Gradient probes** — fixed held-out probe batches (two per topic including core) never
   trained on; measurement only, with no optimizer step and no RNG/parameter/optimizer-state
   effect. Records relative gradient norm `|grad_g| / |theta_g|` per (topic, group), plus
   labeled-vs-unlabeled gradient cosine similarity restricted to module `i`'s parameters, the
   same for core, and a mismatched-module control.
3. **Stage 4, wrong labels** — false negatives (5/20/50% of auxiliary stories labelled core),
   false positives (1/5/20% of core stories labelled a random auxiliary), wrong module
   (10/30% of auxiliary stories labelled a different auxiliary), and the 20%-miss comparison of
   "labelled core" versus "unlabeled". Entirely this project's own; no paper counterpart.
4. **Labelled-fraction sweep beyond 50%** — see §3.2.
5. **Re-running evaluation from the saved checkpoint before reporting any number** — a
   verification step the paper does not describe.

---

## 5. Open questions blocking Stage 2's primary condition

1. §1.1 core width 1856 (paper) vs 2048 (what produced Table 1)
2. §1.2 equal steps (paper App. A / brief) vs λ-rescaled length (paper App. I / code)
3. §1.3 weight decay 0.1 (paper) vs 0.01 (code default, used for Table 1)
4. §1.4 elicitation 128 seq / 75 steps (paper) vs 512 seq / early-stopping (code)

None block Stage 1: all four are config options in the implementation. They decide which
configuration is declared primary for the Table 1 comparison.
