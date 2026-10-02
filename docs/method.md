# Method notes

The associated paper is [Lu et al., IEEE JBHI (2025)](https://doi.org/10.1109/JBHI.2025.3556775). The research implementation is in [`code/`](../code/README.md).

## Source map

- Annotation and eligible intervals: annotation helpers and `get_inter_pre_zones` in `data.py`
- Segmentation and splits: record/segment helpers and `split_segment_base` in `data.py`
- Coherence: `compute_coherence` and cached `EEGDataset` access in `data.py`
- Architecture and training: `models.py`, `train.py`, `config.py`
- Predictions and thresholds: `evaluate.py`, `util.py`
- Warning intervals and seizure-level evaluation: `warning.py`

## Important choices

The current code includes exploratory configurations. Select an explicit configuration for the experiment you intend to run.

- Current CNN widths are `[512, 256, 128, 64]` and learning rate is `1e-6`; the paper's CNN widths are `[32, 64, 128, 256]` with learning rate `1e-5`
- Current training uses AdamW and a mixture of BCE and a differentiable performance-product proxy (`LOSS_PP_WEIGHT=0.5`). Validation/checkpoint selection uses the proxy alone
- Required preprocessing keys, prepared artifacts and channel-pair ordering need to match the selected data path. See the [setup guide](setup.md)
- `val`, `tst`, `dynamic` and `fixed` threshold routes are distinct. `tst` is a test-selected oracle condition. Keep threshold-selection provenance with each output
- Some smoothing paths operate on logits, and `smooth_logit_per_sz` includes a centered window. Review timing before interpreting a selected path as causal probability smoothing
- Warning metrics depend on SPH/SOP definitions, retriggering, exclusions and time denominators. The current validation-selected warning branch needs the correction noted in the setup guide

The optional [`paper_reference/`](../paper_reference/README.md) directory provides separate numerical components with synthetic tests. It is supplementary to the research source and documents its own conventions.
