# Project spec: iterative-error-feedback refinement head for uncertainty-aware gaze

## Goal
Add a small, learned **refinement head** on top of my existing L2CS-Net gaze pipeline. Starting from the raw L2CS prediction, it iteratively predicts a small *correction* to the gaze distribution and applies it a few times, so the final estimate has a **more accurate mean** and a **better-calibrated spread** (a right-sized uncertainty cone — tight when confident, wide when ambiguous).

This is the gaze analog of Carreira et al., "Human Pose Estimation with Iterative Error Feedback" (CVPR 2016, arXiv:1507.06550). Read that idea as the conceptual reference: don't predict the answer in one shot; feed the current estimate back in and predict a bounded correction, repeated a few times.

## Important: the goal is NOT "make sigma smaller"
A naive shrink of the spread just produces an overconfident, miscalibrated model. Improvement must be demonstrated by **calibration**, not by a smaller sigma. Every experiment must report coverage alongside angular error (see Evaluation). Treat this as a hard requirement, not a nice-to-have.

## My environment
- Existing codebase: L2CS-Net (ResNet-50 backbone, full-face input). Two heads (yaw, pitch); each outputs logits over angle bins; predicted angle = softmax then expected value (soft-argmax); trained with combined cross-entropy-on-bins + MSE-on-angle loss.
- Hardware: RTX 4060 Laptop, 8 GB VRAM. Be memory-conscious: freeze the backbone during refinement-head training, use modest batch sizes, allow gradient accumulation if needed.
- PyTorch. Python virtual env. Single GPU.
- Do NOT assume I am an expert in L2CS internals — explain non-obvious choices briefly in comments and in your messages to me.

## The new module: refinement head `f`
At each refinement step `t`:
- **Inputs:** (a) the backbone's **spatial feature map** (before global pooling — this is the new information the original head discards), and (b) the **current gaze estimate** encoded as a small vector (current mean yaw/pitch plus a summary of the current distribution such as its variance/entropy; optionally the full per-bin probabilities).
- **Mechanism:** a lightweight head (start with the simplest version, then upgrade — see Build order) that conditions on the current estimate and attends to / convolves over the spatial feature map.
- **Output:** a correction `ε_t` to the bin logits for yaw and pitch (a residual).
- **Update:** `logits_{t+1} = logits_t + ε_t`, then recompute the distribution and its soft-argmax mean.
- **Bounded step:** cap how far the mean angle can move in one step (this is the analog of the 20-pixel cap in the IEF paper). Make the cap a config value.

The first estimate (`t=0`) is the raw L2CS output: run the base model once, then loop only the small head.

## Training
- **Freeze** the backbone and the original L2CS heads initially; train only the refinement head. Leave a config flag to optionally unfreeze and fine-tune later.
- **Bounded-step curriculum (do not skip this):** do not train each step to jump all the way to ground truth. Build a fixed schedule of intermediate targets, each a bounded step from the current estimate toward the true gaze. The IEF paper showed that removing this causes ~10-point degradation and drift over steps. Make the schedule explicit and documented.
- **Per-step loss:** negative log-likelihood of the true gaze under the step's predicted distribution (or cross-entropy on bins consistent with how L2CS is trained), summed over the refinement steps following the curriculum. Add a calibration-oriented regularizer so the spread is encouraged to match the residual error rather than collapse.
- **Steps:** default 3–4 refinement iterations; make it configurable.
- Log per-step metrics every epoch so we can see the estimate converge.

## Evaluation (must-have, against the raw L2CS baseline)
1. **Mean angular error** (degrees), final vs baseline.
2. **Calibration:**
   - Coverage probability — does the predicted 95% cone actually contain ground truth ~95% of the time? (Reference: Zheng et al. 2025, arXiv:2501.14894, coverage-based proper metric.)
   - Reliability diagram and expected calibration error (ECE) and/or NLL.
3. **Per-step curves:** plot angular error AND coverage as a function of refinement step (like the PCKh-vs-step figure in the IEF paper). This is the headline figure: it should show error dropping and coverage staying honest across steps.
4. **Ablations:**
   - with vs without the bounded-step curriculum,
   - with vs without spatial-feature-map conditioning (i.e. does the extra info actually help, or is the head just relearning the base output?),
   - iterative (3–4 steps) vs single-step correction.

## Build order (do these as separate, reviewable stages — show me a plan before coding each)
1. Scan the repo. Wrap the existing L2CS model to expose (a) the spatial feature map and (b) per-axis bin logits, without changing its current behaviour. Add a quick test that the wrapped model reproduces the original predictions.
2. Dataloader that yields (image, ground-truth yaw/pitch) and the cached base L2CS prediction for each sample.
3. Refinement head — simplest version first (an MLP on the pooled feature + estimate vector). Get the full loop training end-to-end before adding spatial attention.
4. Training loop with the bounded-step curriculum and NLL loss, backbone frozen. Per-step logging.
5. Evaluation harness: angular error + coverage + reliability diagram + per-step curves.
6. Upgrade the head to attend over the spatial feature map; rerun.
7. Ablations.

## Guardrails for you (Claude Code)
- Never report a result as an improvement based on a smaller spread alone — always pair it with coverage.
- Keep VRAM usage in check (frozen backbone, small batches); tell me if something risks OOM on 8 GB.
- Make steps, bound, and loss weights configurable; don't hardcode.
- Prefer small, testable increments; after each stage, run on a tiny subset first and show me the numbers before scaling up.
