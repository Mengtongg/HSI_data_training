# Hyperspectral Imaging Tissue Classification

This repository contains the code used for tissue classification using hyperspectral imaging (HSI) spectra.

The goal of the project is to evaluate how well different organs can be distinguished based on their spectral signatures, and to investigate how preprocessing and dataset construction affect classification performance.

Two types of datasets are used:

1. **TXT spectra dataset**
   - Manually selected spectra exported from MATLAB HSI viewer.
   - Represents a clean baseline dataset.

2. **H5 hyperspectral cube dataset**
   - Full hyperspectral cubes.
   - Spectra are extracted automatically from pixel sampling.

The project compares these pipelines and investigates how preprocessing, pixel sampling strategies, and frequency selection affect classification performance.

---

# Evaluation protocol

All experiments use **leave-one-day-out evaluation**.


- Train on Day 2 + Day 3 → Test on Day 1
- Train on Day 1 + Day 3 → Test on Day 2
- Train on Day 1 + Day 2 → Test on Day 3

This avoids pixel-level data leakage and evaluates cross-day generalisation.

Metrics reported:

- Spectrum-level accuracy
- Macro F1 score
- File-level accuracy

File-level accuracy aggregates predictions across spectra extracted from the same HSI cube.

---

# Repository structure
```
HSI_data_training/
├── artifacts/
|
├──configs/
|  └── paths.py
|
├──metadata/
|  ├── metadata.csv
|  └── metadata_h5.csv
│
├── src/
|
|   Dataset construction
│   ├── build_dataset.py
│   ├── build_h5_dataset.py
|   └── loader.py
│       
│   Baseline training
|   └── train_1.py
|
│   H5 experiment
|   ├── train_h5_1.py
|   ├── train_h5_2.py
|   ├── train_h5_3.py
│   └── train_h5_4.py
|
|   Data inspection
|   ├── inspect_h5.py
|   ├── view_h5.py
|   ├── extract_h5_spectra.py
|   └── check_h5_dataset.py
|
|   Analysis and visualisation
|   ├── plot_mean_spectra.py
|   └── plot_h5_pca.py
|
├── pyproject.toml
├── requirements.txt
└── README.md
```


---

# File descriptions

## Dataset construction

### build_dataset.py

Builds a classification dataset from manually selected spectra stored as TXT files.

Main steps:

- read spectra from TXT files
- align them to a common frequency grid
- assign labels based on metadata
- save arrays for training

Output includes:
```
X.npy
y.npy
days.npy
groups.npy
freq.npy
```

---

### build_h5_dataset.py

Constructs a dataset from full hyperspectral cubes (.h5).

Processing steps:

1. read hyperspectral cube
2. read frequency axis
3. compute intensity image
4. apply outer tissue mask using threshold
5. optionally apply inner mask
6. randomly sample pixels
7. interpolate spectra to a common frequency grid
8. save dataset arrays

Output stored in:
```
artifacts/training_h5_2/
```

---

# Baseline model

### train_1.py

Baseline classifier trained using the **TXT spectra dataset**.

Classifier:
- Logistic Regression

Evaluation:
- leave-one-day-out cross validation

Outputs:

- classification report
- macro F1 score
- spectrum accuracy
- file level accuracy

---

# H5 experiment pipeline

Several scripts explore different dataset construction strategies.

---

## train_h5_1.py

Initial H5 experiment.

Features:

- extract spectra from H5 cubes
- no frequency filtering
- simple pixel sampling
- logistic regression classifier

Purpose:

Provide the first baseline using H5 data.

---

## train_h5_2.py

Investigates dataset construction parameters.

Parameters tested:

- tissue threshold
- inner mask value
- number of sampled pixels

Purpose:

Evaluate how pixel and threshold selection affects classification performance.

---

## train_h5_3.py  (Final best performing pipeline)

This is the final pipeline producing the best performance.

Key features:

- optimized threshold
- optimized inner mask
- optimized pixel sampling
- frequency range selection
- standard scaling preprocessing

Classifier:

Logistic Regression

Evaluation:

Leave-one-day-out validation.

This script should be used to reproduce the main results of the project.

---

## train_h5_4.py

Additional preprocessing experiments.

Includes testing of:

- SVM classifier
- SNV (Standard Normal Variate)
- area normalization
- other spectral normalization methods
- PCA
- SG smoothing

Observation:

These preprocessing methods **did not improve performance compared with train_h5_3**.

Therefore train_h5_4 is included mainly for method comparison and completeness.

---

# Analysis tools

### plot_mean_spectra.py

Plots the mean spectrum of each organ.

Purpose:

Visual inspection of spectral differences between organs.

Used to guide frequency range experiments.

---

### plot_h5_pca.py

Performs PCA on sampled spectra and visualizes data distribution.

Outputs:

- PC1 vs PC2 scatter plot
- PCA loading plots

Purpose:

- understand spectral distribution
- detect overlap between organs
- inspect cross-day domain shift

---

### inspect_h5.py

Quick inspection tool for H5 files.

Prints:

- dataset keys
- cube shape
- frequency axis
- saturation map information

---

### view_h5.py

Visualises H5 cube components:

- band images
- intensity image
- saturation map

Useful for debugging.

---

### extract_h5_spectra.py

Extracts spectra from H5 cubes for inspection.

Used to check:

- pixel mask
- number of valid spectra
- spectral shapes.

---

### check_h5_dataset.py

Checks consistency of the generated H5 dataset:

- shape
- label distribution
- frequency axis alignment

---

# Confusion matrices

The training scripts also produce **confusion matrices**.

These visualise which organs are commonly misclassified.

Typical observations:

- anatomically similar organs show higher confusion
- cross-day spectral shift affects classification accuracy.

---

# Installation

Create a virtual environment:
```
python-m venv .venv
```

Activate:
Windows
```
.venv\Scripts\activate
```


Install dependencies:
```
pip install -r requirements.txt
```


---

# Running the final experiment

Run the final pipeline:
```
$env:HSI_THRESH=85
$env:HSI_INNER=40
$env:HSI_PIXELS=2000
$env:HSI_FMIN=333
$env:HSI_FMAX=748
python -m src.train_h5_3
```


This will:

1. load the H5 dataset
2. apply preprocessing
3. train the classifier
4. evaluate leave-one-day-out performance
5. output classification reports and summary metrics

---

# Notes

This project focuses on analysing the effects of dataset construction and preprocessing on HSI spectral classification rather than building deep neural networks.

The number of HSI cubes is relatively small, which makes classical models such as logistic regression more stable than deep learning models in this setting.

