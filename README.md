# Vibration-based Normal vs Failure Detection with STFT + CNN

A reproducible end-to-end pipeline to classify metro bogie condition (Normal vs Failure) from 3‑axis vibration signals using STFT spectrograms and a compact 2D CNN.

- Input: 3-axis accelerometer time series at 1024 Hz
- Transform: STFT → 3‑channel spectrograms (x, y, z)
- Model: Shallow 2D CNN with adaptive average pooling
- Reporting: Per-seed classification reports and confusion matrices (counts only), plus aggregated summaries

Run in Colab:
- Click to open: https://colab.research.google.com/drive/1OdeFevb1rgEUllE7g1PV6OMwJX2xri2Z?usp=sharing
- Notebook to run: Metro_Dataset_2D_Transformation.ipynb

Open in Colab badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OdeFevb1rgEUllE7g1PV6OMwJX2xri2Z?usp=sharing)

## Problem Statement

Metro bogies and track structures experience continuous mechanical stress. Early detection of abnormal behavior reduces downtime, maintenance costs, and safety risks. The goal is to detect whether the bogie condition is Normal or Failure directly from onboard vibration data. We formulate this as a binary classification problem using time–frequency representations (STFT spectrograms) and a 2D CNN.

## Dataset

- Source: MetroDataset (GitHub: https://github.com/EnfangCui/MetroDataset)
- Setup: Two sensor nodes on different bogies: one Normal, one Failure
- Sensors: 3-axis accelerometer (X/Y/Z), unit g, sampling frequency 1024 Hz
- Duration: ~50.8 min Normal, ~101.4 min Failure
- Files (after download/unzip):
  - Normal/: Metro_vibration_v1_[x|y|z]_axis_normal.csv
  - Failure/: Metro_vibration_v1_[x|y|z]_axis_failure.csv

Note: Labels reflect bogie condition; track-induced vibrations can also influence signals.

## Approach

1) Windowing and Overlap
- Segment each axis into 20 s windows (20 × 1024 = 20,480 samples)
- Overlap controlled by STRIDE = SEGMENT_LENGTH × (1 − overlap)
  - 80% overlap → STRIDE = 0.2 × window_length
  - 60% overlap → STRIDE = 0.4 × window_length
  - 50% overlap → STRIDE = 0.5 × window_length

2) STFT Transformation
- PyTorch STFT with n_fft=256, hop_length=128, center=True
- Output per axis: magnitude spectrogram with shape (freq_bins=129, time_frames=161)
- Stack x/y/z → (H, W, C) = (129, 161, 3); transpose to (C, H, W) for PyTorch

3) Model
- Shallow CNN:
  - 4 conv layers with ReLU, 3 × MaxPool2d(2)
  - AdaptiveAvgPool2d((1, 1)) → Flatten → Linear(128→64→2)
- Parameters: 105,826 trainable

4) Training & Evaluation
- Random 80/20 stratified split at the segment level (per seed)
- Seeds: [100, 500, 750, 1000, 1200]
- Loss: CrossEntropyLoss; Optimizer: Adam (LR=1e-3)
- Reporting:
  - Per-seed classification reports and confusion matrices (counts only)
  - Aggregated accuracy/loss summaries across seeds

## Repository Contents

- Metro_Dataset_2D_Transformation.ipynb — Main notebook (run this)
- Generated outputs (auto-created after running):
  - Processed_STFT_Transform/ — Saved spectrogram segments (.npy)
  - Reports/exports/ — CSVs (per-seed results, summary tables, classification reports)
  - Reports/plots_cm/ — Confusion matrix images (counts only) per seed

## Environment

- Python 3.8–3.11 recommended
- Dependencies:
  - torch, torchvision (match CUDA/CPU to your system)
  - numpy, pandas, tqdm
  - scikit-learn, matplotlib, seaborn
  - IPython (for notebook display)

Install (example):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # pick CUDA/CPU per your setup
pip install numpy pandas tqdm scikit-learn matplotlib seaborn ipython
```

## How to Run (by Notebook Cells)

Open Metro_Dataset_2D_Transformation.ipynb and run top-to-bottom. You can toggle overlap and epochs in the configuration cell.

- Cell 0 — Download & Unzip Dataset
  - Clones MetroDataset from GitHub
  - Recursively unzips any .zip files
  - Renames folder to MetroDataset-master (expected by later cells)

- Cells 1–2 — Setup & Configuration
  - Imports, paths, hyperparameters
  - Key knobs:
    - STRIDE (overlap):
      - 80%: STRIDE = int(SEGMENT_LENGTH * 0.2)
      - 60%: STRIDE = int(SEGMENT_LENGTH * 0.4)
      - 50%: STRIDE = int(SEGMENT_LENGTH * 0.5)
    - EPOCHS: 30, 50, 100 (as per experiments)
    - SEEDS: [100, 500, 750, 1000, 1200]

- Cells 3–4 — Utilities & STFT Preprocessing
  - Loads CSVs, segments signals, computes STFT magnitudes
  - Saves stacked spectrograms to:
    - ./Processed_STFT_Transform/STFT/Normal
    - ./Processed_STFT_Transform/STFT/Failure

- Cells 5–6 — Dataset & Model
  - Loads saved .npy spectrograms and defines ShallowCNN
  - CNN Input: (C, H, W) = (3, 129, 161); Total params ≈ 105,826

- Cell 7 — Model Summary (Tabular)
  - Prints layer-by-layer output shapes and parameter counts

- Cell 8 — Train & Eval Helpers
  - Train loop and evaluation functions

- Cell 9 — Run & Report
  - Trains/evaluates across SEEDS
  - Prints per-seed classification report
  - Saves:
    - Per-seed and summary CSVs → ./Reports/exports/
    - Confusion matrices (counts only) → ./Reports/plots_cm/

## What Gets Created (Folder Structure)

After running the notebook, you’ll typically see:

```
.
├── Metro_Dataset_2D_Transformation.ipynb
├── MetroDataset-master/
│   ├── Normal/
│   │   ├── Metro_vibration_v1_x_axis_normal.csv
│   │   ├── Metro_vibration_v1_y_axis_normal.csv
│   │   ├── Metro_vibration_v1_z_axis_normal.csv
│   │   └── README.md
│   ├── Failure/
│   │   ├── Metro_vibration_v1_x_axis_failure.csv
│   │   ├── Metro_vibration_v1_y_axis_failure.csv
│   │   ├── Metro_vibration_v1_z_axis_failure.csv
│   │   └── README.md
│   └── README.md
├── Processed_STFT_Transform/
│   └── STFT/
│       ├── Normal/
│       │   ├── segment_00000.npy
│       │   ├── segment_00001.npy
│       │   └── ...
│       └── Failure/
│           ├── segment_00000.npy
│           ├── segment_00001.npy
│           └── ...
└── Reports/
    ├── exports/
    │   ├── STFT_per_seed_results.csv
    │   ├── STFT_summary_accuracy_loss.csv
    │   ├── STFT_summary_additional_metrics.csv
    │   ├── STFT_seed100_classification_report.csv
    │   ├── STFT_seed500_classification_report.csv
    │   ├── STFT_seed750_classification_report.csv
    │   ├── STFT_seed1000_classification_report.csv
    │   └── STFT_seed1200_classification_report.csv
    └── plots_cm/
        ├── STFT_seed100_cm_counts.png
        ├── STFT_seed500_cm_counts.png
        ├── STFT_seed750_cm_counts.png
        ├── STFT_seed1000_cm_counts.png
        └── STFT_seed1200_cm_counts.png
```

Notes:
- If you change overlap (STRIDE) or EPOCHS and want to regenerate spectrograms, delete the corresponding folder under ./Processed_STFT_Transform/STFT/ and rerun Cells 3–4.
- The notebook will auto-create missing folders (Processed_STFT_Transform, Reports/exports, Reports/plots_cm).

## Results (Example: 80% overlap, 50 epochs)

Per-seed (from a representative run):
| Seed | Accuracy | Loss   | TN  | FP | FN | TP  |
|------|----------|--------|-----|----|----|-----|
| 100  | 0.9896   | 0.0319 | 177 | 0  | 5  | 298 |
| 500  | 0.9875   | 0.0377 | 175 | 2  | 4  | 299 |
| 750  | 0.9896   | 0.0377 | 175 | 2  | 3  | 300 |
| 1000 | 0.9979   | 0.0111 | 177 | 0  | 1  | 302 |
| 1200 | 0.9792   | 0.0470 | 175 | 2  | 8  | 295 |

Across seeds:
- Accuracy (mean ± std): 0.98875 ± 0.00669
- Loss (mean ± std): 0.03307 ± 0.01343
- F1 Macro Mean: 0.9880; F1 Weighted Mean: 0.9888
- Class-wise means:
  - Normal: Precision 0.9768, Recall 0.9932, F1 0.9849
  - Failure: Precision 0.9960, Recall 0.9861, F1 0.9910

Artifacts saved:
- CSVs: ./Reports/exports/
- Confusion matrices (counts only): ./Reports/plots_cm/

## Notes, Limitations, and Tips

- Evaluation note:
  - With heavy overlap and random segment-level splits, near-duplicate segments can end up in both train and test, inflating accuracy.
  - For stricter evaluation, split by time before segmenting and keep test windows non-overlapping.
- Re-running experiments:
  - Change STRIDE (overlap) and EPOCHS in the configuration cell; rerun Cell 4 to regenerate STFT, then Cell 9 to train/evaluate.
- GPU:
  - The notebook auto-detects GPU (if available). For CPU-only, it still works (just slower).

## Future Work

- Preprocessing:
  - Hann window + log-power compression; frequency cropping (e.g., 4–250 Hz)
  - Per-channel/per-frequency normalization using training statistics
- Modeling:
  - 1D CNN/TCN baselines; spectrogram augmentations; attention modules
- Evaluation:
  - Time-based splits; blocked cross-validation with margins; non-overlapping test
- Deployment:
  - Sliding-window real-time inference; edge device profiling

## Acknowledgements

- Dataset: MetroDataset by EnfangCui et al. (https://github.com/EnfangCui/MetroDataset)
- Libraries: PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
--- 

