# Research source

The research workflow consists of these APIs:

1. Configure private inputs and experiment settings in `config.py` through environment variables and private JSON
2. Prepare annotation/segment manifests and coherence caches with the helpers in `data.py`
3. Train using `train.train` and the architectures in `models.py`
4. Produce saved-model predictions and metrics using `evaluate.py`
5. Analyze warning behavior using `warning.py`

See the [setup guide](../docs/setup.md) for the input contract and signatures. Modules use sibling imports, so run research API calls from `code/` or add it to `PYTHONPATH`.

## Modules

- `config.py`, `experiment_settings.py`, `runtime_paths.py`: experiment settings and private paths
- `data.py`: annotations, segments, splits, coherence calculation and cached dataset access
- `models.py`, `train.py`: CoSP CNN, exploratory architectures, training and checkpoints
- `evaluate.py`, `warning.py`, `util.py`: inference, thresholds, metrics, warnings and shared helpers
- `analysis.py`: figures, score comparisons and coherence analysis
- `explanation.py`, `explain_tabular.py`, `connectivity.py`: exploratory explanation and connectivity analyses
- `requirements.txt`: historical dependency pins

Data-dependent analysis functions accept private inputs. For example, `wilcox_test(cosp_scores, compare_scores)` takes aligned score arrays; `plot_results_histogram(data)` takes a caller-supplied metric table; `weight_compare(pre_path, inter_path, ictal_path)` takes weight files. `evaluate_methods_all(pids, pre_itvls, ...)` requires an explicit selection. Image-based functions accept electrode/image paths or use the private `xray_files` mapping. No participant result values are embedded in these helpers.

Training, inference and analysis can write private outputs. The experiment-only direct-execution launchers are omitted; select the appropriate API and settings for your experiment. Current source defaults include exploratory choices described in the [method notes](../docs/method.md).
