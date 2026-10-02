# Setup and private inputs

## Environment

Create an isolated environment and review `code/requirements.txt`. Its historical pins combine PyTorch 1.7.1 with newer scientific packages; choose a compatible Python/PyTorch/CUDA combination and retain the resolved environment for your experiment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python -m pip check
```

Core modules use NumPy, pandas, SciPy, h5py, joblib, matplotlib, scikit-learn and torch. Optional modules also require:

- ViT: `timm`; architecture summaries: `torchinfo`
- Explanation: `lime`, `scikit-image`, `seaborn`
- Warning analysis: `seaborn`, `imbalanced-learn`
- Connectivity: `networkx`, `python-louvain` (imported as `community`)

The additional dependencies are not all pinned in the historical list. The MATLAB loader also invokes the operating-system `file` command.

## Private configuration

Set these variables before importing a research module:

```bash
export COSP_DATA_PATH=/absolute/path/to/authorized/recordings
export COSP_WORK_PATH=/absolute/path/to/private/experiment
export COSP_FILE_LEN_MIN=<record_duration_in_minutes>
export COSP_REC_OFFSET=<truncated_sample_offset>
export COSP_EXPERIMENT_CONFIG=/absolute/path/to/private/experiment.json
```

Replace the angle-bracket placeholders with integers appropriate to your files. The data path must exist; both paths must be absolute. Importing `config` creates output directories and may select a CUDA device. Do not run concurrent jobs using the same work directory and run identifiers.

Copy the empty [configuration template](../examples/experiment.example.json) outside the checkout. It has these fields:

- `participants`: ordered objects with `pat` (recording-directory identifier) and `days` (positive recording duration). Used by participant-summary analysis
- `final_info`: objects with integer `pat_id` and string `pat`. Optional `model4`, `model15`, `model30`, `model45`, `model60`, `model90`, `model120`, `model150`, `model180` and `thd45_opt` fields reference your own saved models/thresholds. Extra legacy mapping columns are retained for helpers that need them
- `pid_xray`: participant IDs selected for optional image analysis
- `xray_files`: participant-ID strings mapped to objects with `electrodes` (MAT path containing `ELoc`) and `image` (image path)
- `config`: overrides for `CONFIG` in `code/config.py`

For training, set `config.pat_id` to an ID in `final_info`. Also provide the preprocessing fields `freq`, `n_chn`, `rec_len_min` and `detrend` from your intended protocol; constants elsewhere do not populate them automatically. Review architecture, segmentation, sample limits, split fractions and loss settings before importing data modules, since some defaults capture configuration at import time. Older preparation functions also read `config.patient`; set it consistently with the selected recording identifier when using those paths.

The empty template allows metadata helpers to load without inventing a cohort. A scientific run requires actual authorized private metadata, settings and inputs.

## Prepared data layout

```text
recordings/
  Annotations/
  Patient_<participant>/Data_<YYYY>_<MM>_<DD>/Hour_<HH>/
    UTC_<HH>_<MM>_<SS>.mat
private/experiment/
  datainfo/
    annotation/<participant>_Annots.csv
    annotation/pat_info.csv
    annotation/record_start.csv
    annotation/record_lengths.csv
    record_list/
    segment_list/
  data/<participant>/
    <participant>_sz_id_used.csv
    <coherence-cache>.pkl
  model/
  result/detail/
```

Some readers also recognize `CUTC_` filenames. Annotation helpers expect fields including `sz_epoch`, `pat_start_epoch`, `sz_id`, `lead_12` and `sz_4h`; other preparation paths require additional fields.

`split_segment_base` reads eligible seizure IDs from the private seizure-ID file, sorts them and splits using the configured fractions. Check ordering and eligibility against the intended chronological protocol.

The active `EEGDataset` reads trusted coherence-cache dictionaries keyed by `seg_epoch`. Values are tensor-like frequency-by-channel-pair arrays, normally `[109, 120]`. Confirm pair ordering against the cache producer. Missing entries advance to a later sample, which can duplicate samples or raise `IndexError`; verify full cache coverage before use. The active loader does not compute missing features from raw EEG.

Some older preparation helpers need adaptation: `compute_coh_pat` uses an older batch contract; the fresh non-overlap base lacks a seconds-to-seizure field read by its cached path; the MATLAB annotation converter calls `to_csv` on an array. Prepared artifacts must match the reader you use. Load only trusted pickle files and checkpoints.

## Execution APIs

Call these with `code/` on the import path after configuration:

- `train.train(rank=None)`: `None` selects CPU; integer ranks select CUDA paths. Writes training artifacts and normally returns `{'model_id': ..., 'round_id': ...}`; an existing training marker may cause a skipped run and `None`
- `evaluate.test(trn_model_ids, model_type='CoSP', rank=None, data_type='tst', inter_itvl=1440, save=True)`: supply the matching model/round IDs, architecture, private `result/result.csv` metadata and checkpoint. Set the device explicitly, for example `rank='cpu'`. Use `data_type='val'` for validation predictions
- `evaluate.evaluate_result(model_id, thd_from='val', thd_fixed=None, pre_itvl=45, inter_itvl=1440, pat_id=None, smooth_min=0)`: choose the threshold source explicitly

Inference returns `None` and writes prediction CSVs even when `save=False`. With `data_type='tst'` and `save=True`, it also computes test-selected (`thd_from='tst'`) oracle metrics. Validation-threshold selection reads a validation file with a fixed 1440-minute interval. Training/inference skip singleton batches; evaluation removes duplicate epochs and seizure groups without positive segments. Retain those counts in your private experiment record.

For warnings, select smoothing domain, warning length, retriggering, time denominator and threshold policy explicitly. The validation-selected branch in `evaluate_ma_pre_itval` currently does not append test metrics; address that path before relying on it.

## Checks and experiment records

From the repository root:

```bash
python3 tools/check_readiness.py --syntax-only
python3 -m unittest discover -s tests -v
python3 tools/check_readiness.py --config /absolute/path/to/private/experiment.json
```

The final command checks required preprocessing keys without importing research modules. It exits with status 1 if they are absent. Keep the effective configuration, code revision, environment, input/split/cache provenance, random seeds, checkpoint-selection rule and evaluation protocol with your private outputs.
