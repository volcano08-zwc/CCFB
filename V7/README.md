> Local experiment V7: V5 plus AdamW, five-epoch warmup, and cosine learning-rate decay. See `EXPERIMENT_CHANGE.md`.

<div align="center">
<h1>DBConformer</h1>
<h3>Dual-Branch Convolutional Transformer for EEG Decoding</h3>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1</sup>, [Hongbin Wang](https://github.com/WangHongbinary)<sup>1</sup>, [Tianwang Jia](https://github.com/TianwangJia)<sup>1</sup>, [Xingyi He](https://github.com/BAY040210)<sup>1</sup>, [Siyang Li](https://scholar.google.com/citations?user=5GFZxIkAAAAJ&hl=en)<sup>1</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

(<sup>:email:</sup>) Corresponding Author

[![DBConformer](https://img.shields.io/badge/Paper-DBConformer-2b9348.svg?logo=IEEE)](https://ieeexplore.ieee.org/document/11215634)&nbsp;
[![Supplementary](https://img.shields.io/badge/Supplementary-DBConformer-2b9348.svg?logo=IEEE)](https://drive.google.com/file/d/1BswpJ8kJ_laORqA63xPCxhWpetxT9-3u/view?usp=sharing)&nbsp;

</div>

> This is a streamlined copy of the implementation for [**"DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding"**](https://ieeexplore.ieee.org/document/11215634). It contains only the DBConformer model and the Leave-One-Subject-Out (LOSO) experiment.

> **Local V5 experiment:** strict V0-based single-factor ablation. It preserves
> the complete V0 competitive-Softmax model and adds only source-subject
> distribution perturbation after source-subject EA. See
> `EXPERIMENT_CHANGE.md` and run
> `python .\smoke_test_source_subject_perturbation.py` before full training.

## Overview

📰 News: **DBConformer** has been accepted for publication in the [**IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)**](https://ieeexplore.ieee.org/document/11215634). Congratulations! 🎉

📰 News: The original EEG trials of our seizure dataset CHSZ are now publicly available on [Zenodo](https://zenodo.org/records/19333249).

📰 News: We've released the [supplementary material](https://drive.google.com/file/d/1BswpJ8kJ_laORqA63xPCxhWpetxT9-3u/view?usp=sharing) for DBConformer.

📰 News! We've reproduced and added three recent EEG decoding baseline models, including [MSVTNet](https://ieeexplore.ieee.org/abstract/document/10652246), [MSCFormer](https://www.nature.com/articles/s41598-025-96611-5), and [TMSA-Net](https://www.sciencedirect.com/science/article/pii/S1746809424012473).

**DBConformer**, a **dual-branch convolutional Transformer** network tailored for EEG decoding:

- **T-Conformer**: Captures temporal dependencies
- **S-Conformer**: Models spatial patterns
- A **lightweight channel attention module** further refines spatial representations by assigning data-driven importance to EEG channels

<div align="center">
<img width="1120" height="477" alt="image" src="https://github.com/user-attachments/assets/71de7f6e-3dcc-4deb-9a00-382235081111" />
</div>

## Features

- 🔀 **Dual-branch parallel design** for symmetric spatio-temporal modeling
- 🧩 **Plug-and-play channel attention** for data-driven channel weighting
- 📈 **Strong generalization** across CO, CV, and LOSO settings
- 💡 **Interpretable** aligned well with sensorimotor priors in MI
- 🧮 8× fewer parameters than large CNN-Transformer baselines (e.g., EEG Conformer)

Comparison of network architectures among CNNs (EEGNet, SCNN, DCNN, etc), traditional serial Conformers (EEG Conformer, CTNet, etc), and the proposed DBConformer. DBConformer has two branches that parallel capture temporal and spatial characteristics.

<div align="center">
  <img width="553" height="453" alt="image" src="https://github.com/user-attachments/assets/f819600d-27de-460e-a82a-4f7e5322e168" />
</div>

## Code Structure
```
DBConformer/
│
├── DBConformer_LOSO.py     # Main script for Leave-One-Subject-Out (LOSO) scenario
│
├── models/
│   └── DBConformer.py      # Dual-branch Convolutional Transformer
│
├── data/                   # Dataset
│   ├── BNCI2014001/
│   └── ...
│
├── utils/                  # Helper functions and common utilities
│   ├── data_utils.py           # EEG preprocessing, etc
│   ├── alg_utils.py           # Euclidean Alignment, etc
│   ├── network.py        # Backbone definition
│   └── ...
│
└── README.md
```

## Scope
This copy intentionally excludes the CO and CV experiment scripts and all comparison-model implementations. The retained training entry point always uses DBConformer in the LOSO setting.

## Datasets
DBConformer is evaluated on **MI classification** and **seizure detection** tasks. MI datasets can be downloaded from [MOABB](https://moabb.neurotechx.com), and [NICU dataset](https://zenodo.org/record/4940267). The processed BNCI2014001 dataset can be found in [MVCNet](https://github.com/wzwvv/MVCNet).

- Motor Imagery:
  - BNCI2014001
  - BNCI2014004
  - Zhou2016
  - Blankertz2007
  - BNCI2014002
- Seizure Detection:
  - CHSZ
  - NICU

## Experimental Scenario

- **LOSO (Leave-One-Subject-Out):** Cross-subject generalization evaluation. EEG trials from one subject were reserved for testing, while all other subjects’ trials were combined for training.

## Visualizations
### Effect of Dual-Branch Modeling
To further evaluate the impact of dual-branch architecture, we conducted feature visualization experiments using t-SNE. Features extracted by T-Conformer (temporal branch only) and DBConformer (dual-branch) were compared on four MI datasets.

<img width="1387" alt="image" src="https://github.com/user-attachments/assets/d6d8a8eb-bdf5-4b69-9dca-459f46d9cb8c" />

### Visualization of Spatio-Temporal Self-Attention
To further examine the interpretability of DBConformer, we visualized the self-attention matrices learned in both temporal and spatial branches on BNCI2014001, BNCI2014002, and OpenBMI datasets.

<img width="914" height="741" alt="image" src="https://github.com/user-attachments/assets/4e610689-2b74-4de6-b17e-66ac705e0762" />

### Interpretability of Channel Attention
To investigate the interpretability of the proposed channel attention module, we visualized the attention scores assigned to each EEG channel across 32 trials (a batch) from four MI datasets. BNCI2014004 were excluded from this analysis, as it only contains C3, Cz, and C4 channels and therefore lacks spatial coverage for attention comparison.

<img width="1384" alt="image" src="https://github.com/user-attachments/assets/efebf73d-ea1c-46a8-8287-e5a2a0d352a7" />

### Sensitivity Analysis on Architectural Design
We further conducted a sensitivity analysis to explore how architectural design affects the DBConformer performance.

<img width="1319" height="304" alt="image" src="https://github.com/user-attachments/assets/11e78ef8-d610-438c-8357-391c86c38fb5" />

---

## 📄 Citation
If you find this work helpful, please consider citing our paper:
```
@Article{wang2025dbconformer,
  author  = {Ziwei Wang and Hongbin Wang and Tianwang Jia and Xingyi He and Siyang Li and Dongrui Wu},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  title   = {{DBConformer}: Dual-branch convolutional {Transformer} for {EEG} decoding},
  year    = {2026},
  number  = {5},
  pages   = {4134--4147},
  volume  = {30},
}

```

## License

This project is released under the PolyForm Noncommercial License 1.0.0 for academic and non-commercial research use.

Commercial use, including use in commercial products, services, or clinical/commercial deployment, is not permitted without prior written permission from the authors.

If you use this code or DBConformer in your research, please cite our paper. For commercial licensing inquiries, please contact Ziwei Wang at [vivi@hust.edu.cn].

## 🙌 Acknowledgments

Special thanks to the source code of EEG decoding models: [EEGNet](https://github.com/vlawhern/arl-eegmodels), [IFNet](https://github.com/Jiaheng-Wang/IFNet), [EEG Conformer](https://github.com/eeyhsong/EEG-Conformer), [FBCNet](https://github.com/ravikiran-mane/FBCNet), [CTNet](https://github.com/snailpt/CTNet), [ADFCNN](https://github.com/UM-Tao/ADFCNN-MI), [EEGWaveNet](https://github.com/IoBT-VISTEC/EEGWaveNet), [SlimSeiz](https://github.com/guoruilu/SlimSeiz), [MSVTNet](https://ieeexplore.ieee.org/abstract/document/10652246), [MSCFormer](https://www.nature.com/articles/s41598-025-96611-5), and [TMSA-Net](https://www.sciencedirect.com/science/article/pii/S1746809424012473).

We appreciate your interest and patience. Feel free to raise issues or pull requests for questions or improvements.
