# Stage 1: how the implementation was checked

The code is written from scratch. This note records what it was checked against and how,
so the checks can be repeated. `DEVIATIONS.md` section 6 has the one-line summary.

## Against the official implementation

The official repository (`agencyenterprise/modular-pretraining`, commit `d07d62e9`) was
used as the specification. Nothing from it is in this repository except two files of
plain numbers in `configs/` (`official_token_counts.json`, `official_targets.json`).

| What | How | Result |
|---|---|---|
| The model computes the same function | `scripts/crosscheck_official.py`: copy one set of weights into both models; compare logits, and the gradient of every parameter, for the dense model and for GRAM under each routing rule, at core width 2048 and 1856 | Bit-identical (difference 0.0), except the all-modules path at 2e-6, where the official code takes a different einsum code path. A negative control (one weight nudged by 1e-3) is detected. |
| The data is the same | `scripts/prepare_data.py` compares token counts per topic and split | All 96 counts equal: 547,853,673 training tokens, 60,763,919 held out |
| Batches per topic | their loader arithmetic recomputed from our token counts, and our story-level cap | Equal for all 48 topics; 16,544 batches per epoch |
| The routing plan | `tests/test_routing.py` on the pool sizes in their logs; `tests/test_real_data.py` from the real data | All 13 per-rule batch counts of two logged runs (p_cr 0.5 / p_as 0.3 and p_cr 0.2 / p_as 0.5); step counts 16,544 / 15,143 / 16,021 |
| The learning-rate schedule | `tests/test_training.py` | Phase boundaries 1654 / 13236 / 1654 and the logged rates at the first and last step |
| The compute ratio | `scripts/validate_compute_ratio.py`: their raw losses and baseline curves through our code | All 738 published ratios to 4e-10; Table 1 reproduced |

Both scripts need a local clone of the official repository and import nothing from it
beyond two model files (the cross-check) or read only numbers (the compute-ratio check).

## The five required invariants

`tests/test_routing_invariants.py`, `tests/test_model.py`, `tests/test_probes.py`. Every
routing test first gives all optimizers momentum, so that a group stepped by mistake
would visibly move; "unchanged" means bit-identical parameters and optimizer state.

| | Requirement | Test |
|---|---|---|
| (a) | an auxiliary batch never changes other modules' parameters | `test_an_auxiliary_batch_never_changes_other_modules` (4 modules x 2 rules) |
| (b) | a core batch with no module sampled changes no auxiliary parameters | `test_a_core_batch_without_a_module_changes_no_auxiliary_parameters` |
| (c) | the core group is untouched on auxiliary batches where the p_as draw fails | `test_the_core_is_untouched_when_auxiliary_spread_does_not_fire` (4 modules) |
| (d) | ablating all modules gives the same output as a model with only the core MLP | `test_ablating_every_module_equals_a_model_with_only_the_core_mlp` (bit-identical against a separately built dense model) |
| (e) | running the gradient probes leaves parameters, optimizer state and RNG state unchanged | `test_probes_leave_parameters_optimizers_and_random_state_unchanged` (GRAM and dense, float32 and bfloat16), and `test_the_next_training_step_is_identical_with_and_without_a_probe` |

Beyond the five: heterogeneous accumulation is tested directly
(`test_each_group_accumulates_only_the_micro_batches_routed_to_it` compares each group's
accumulated gradient with the sum of the gradients of exactly the micro-batches routed to
it), whole runs with and without probes or validation sweeps end bit-identical, and an
interrupted run resumes to the bit-identical result, in float32 and in bfloat16.

## Do the tests have teeth?

A test that cannot fail proves nothing, so the routing and probe tests were run against
deliberately broken copies of the code. All twelve bugs were caught.

| Deliberate bug | Caught by |
|---|---|
| MLP modules are never frozen | the runtime routing check, via (a) |
| shared layers (embedding, attention, norms, unembedding) are never frozen | the runtime routing check, via (a) |
| nothing is frozen and the runtime check is removed | the accumulation test |
| every active group is stepped whatever the rule says | (c) |
| an ablated module still contributes to the forward pass | (d) |
| the story mask lets a story see the previous one | the mask tests |
| micro-batch losses are not divided by their number | the accumulation test |
| the probe runs on the model itself instead of a copy | (e) |
| the probe consumes random numbers | (e) |
| the probe nudges the real model's parameters | (e) |
| the labeled rule is run on batch A instead of batch B | the cosine test |
| the unlabeled rule activates only the core | the gradient-norm and cosine tests |

The third row is worth a second look. With freezing gone *and* the runtime check gone,
(a), (b) and (c) still pass: at one micro-batch per optimizer step, not stepping a group's
optimizer is enough to leave it alone. Only the accumulation test notices. This is the
experimental confirmation that Appendix H's freezing is inert at the paper's Simple
Stories settings and matters only with gradient accumulation.

## Things found along the way that changed the implementation

* **RMSNorm epsilon depends on the PyTorch version** for a bfloat16 model (2^-7 up to 2.6,
  float32 epsilon from 2.7). Pinned explicitly. Found by testing an assumption rather
  than trusting it; the first guess was wrong.
* **"bfloat16" means the parameters are bfloat16**, not autocast.
* **The official code caps each topic** at the mean topic size of its class.
* **Batches are single-topic** in the official code, including "unlabeled" ones.
* **TF32** was copied from the official setup and then removed: it does nothing for a
  bfloat16 model and would have degraded the float32 gradient probes on a GPU.
* A constant added to every output bias does not change a softmax: the first version of
  the "tampered checkpoint is rejected" test tampered with nothing.
