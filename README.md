# COSP: Coherence-based Seizure Prediction

Research code for **Leveraging Channel Coherence in Long-Term iEEG Data for Seizure Prediction**.

Sha Lu, Lin Liu, Jiuyong Li, Jordan Chambers, Mark J. Cook and David B. Grayden  
*IEEE Journal of Biomedical and Health Informatics*, 29(8), 5541–5548, 2025

[Paper PDF](paper/CoSP_JBHI_2025.pdf) · [Publisher](https://doi.org/10.1109/JBHI.2025.3556775) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/40168220/) · [Citation](CITATION.cff)

## Overview

CoSP combines pairwise channel-coherence features with a convolutional neural network to estimate preictal probability from long-term intracranial EEG. Prediction processing then produces seizure warnings. The research implementation also contains related exploratory architectures and analyses.

## Code

- [`code/`](code/README.md): data preparation, coherence features, models, training, evaluation, warning generation and analysis
- [Setup and private inputs](docs/setup.md): dependencies, configuration and execution APIs
- [Method notes](docs/method.md): paper-to-code map and important implementation choices
- [`paper_reference/`](paper_reference/README.md): optional supplementary numerical components and synthetic tests
- `tools/` and `tests/`: source checks and configuration-helper tests

Start with these data-free checks from the repository root:

```bash
python3 tools/check_readiness.py --syntax-only
python3 -m unittest discover -s tests -v
```

For an experiment, prepare authorized inputs and follow the [setup guide](docs/setup.md). The research APIs require prepared annotations, complete coherence caches and an explicit private experiment configuration.

## Data and use

EEG recordings, participant metadata, detailed experimental results, feature caches and trained checkpoints are not distributed here. Keep these inputs and generated outputs in an authorized private workspace outside the checkout. Related [Epilepsyecosystem dataset context](https://doi.org/10.1093/brain/awy210) describes a different cohort; consult the [provider](https://www.epilepsyecosystem.org/neurovista-trial-1) and its [access terms](https://www.epilepsyecosystem.org/terms-and-conditions).

This is research software, not a clinical decision-making tool.

## Citation

Lu, S., Liu, L., Li, J., Chambers, J., Cook, M. J., & Grayden, D. B. (2025). Leveraging Channel Coherence in Long-Term iEEG Data for Seizure Prediction. *IEEE Journal of Biomedical and Health Informatics, 29*(8), 5541–5548. [doi:10.1109/JBHI.2025.3556775](https://doi.org/10.1109/JBHI.2025.3556775)

## License

The software is available under the [MIT License](LICENSE). The license does not grant rights to the associated paper, datasets, participant records, trained weights or third-party material. See [contribution guidance](CONTRIBUTING.md).

The [paper PDF](paper/README.md) is © 2025 The Authors and licensed separately under CC BY-NC-ND 4.0.
