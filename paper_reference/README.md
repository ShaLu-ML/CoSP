# Optional method components

Supplementary numerical components for [Lu et al., IEEE JBHI (2025)](https://doi.org/10.1109/JBHI.2025.3556775). The primary research source is in [`code/`](../code/README.md).

Run the synthetic component tests from this directory in a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`cosp.method` exposes segment dropout handling/imputation/FIR filtering, coherence features, chronological lead-seizure splitting, causal probability smoothing with retriggerable warnings, and validation-threshold selection through an explicit scoring callback. `cosp.model` exposes the four-block CNN, Adam optimizer at 1e-5, BCE-with-logits loss, and one batch training step. Apply sigmoid to inference logits to obtain probabilities before smoothing. Input to the CNN is `[batch,1,120,109]`; use `torch.from_numpy(features).float()[None,None]` for one coherence matrix and `model.eval()` for inference.

## Method alignment and explicit conventions

- CNN channels are 32/64/128/256, kernels 7/5/3/3, pooling 2x2, dense width 128 and dropout 0.5 (section II-B2). Conv-BN-ReLU order and same padding retain the original implementation. Four pools yield 7x6; the paper's intermediate 15x15 dimension is inconsistent with repeated halving. Default PyTorch weight initialization is used; the paper does not specify initialization.
- Segment filtering accepts only 4000x16 inputs. More than 10% missing in any channel rejects a segment; exactly 10% is retained. The paper does not resolve the equality boundary or channel-versus-total denominator. Median imputation is channel-specific. FIR length is required from the caller; zero-phase filtering is an explicit implementation choice because order/phase are unspecified.
- Coherence uses an explicit Hann/256-point/50%-overlap Welch grid and requires a detrending choice. This retains the legacy 109-bin grid (DC through 168.75 Hz), which differs from the paper's stated 0.5-Hz lower bound. Exact historical spectral settings remain unresolved. Constant-channel coherence raises an error rather than silently producing invalid features.
- Splits are chronological by supplied seizure timestamp, using floor cuts at 60% and 80%. The caller must supply eligible lead seizures after the first-100-day and four-hour exclusions. No participant-specific assumptions or annotations are bundled.
- Warnings smooth **probabilities**, not logits, over a complete SOP window; use strictly greater-than threshold, retrigger and coalesce uninterrupted intervals. Times mean prediction availability, gaps restart smoothing, and intervals are half-open. These boundary conventions are explicit. This function does not score seizures or apply the one-minute SPH.
- Threshold selection maximizes validation SS × (1 − TiW). The caller supplies the grid and a validation-only evaluator; test predictions must never be used to choose it.

## Component boundaries

Data ingestion, lead-seizure/postictal masks, balanced overlapping preictal sampling, the paper's total training-segment cap, epoch/early-stop choices and full SPH-aware seizure metrics belong to the experiment pipeline. These small components do not supply that pipeline or any patient inputs, results or checkpoints.

Software is covered by the root [MIT License](../LICENSE). Dataset and paper rights are separate.
