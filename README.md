# Low Cost Motor Imagery Decoding for Rehab (Single Subject)

<div align="center">
  <video controls autoplay loop muted playsinline width="1000">
    <source src="https://cdn.pixabay.com/video/2023/04/15/159049-818026306_large.mp4" type="video/mp4" />
    <a href="https://pixabay.com/videos/brain-nervous-sci-fi-infinite-159049/">Watch the brain visualization on Pixabay</a>
  </video>
</div>

<p align="center">
  <strong>Single-subject EEG decoding for rehabilitation robotics</strong>
</p>

This project focuses on decoding motor attempt from EEG signals for a single-subject rehabilitation setting. The goal is to build a robust machine learning or deep learning pipeline that can distinguish between rest and movement intent using low-cost EEG data.

## Overview

This repository is designed for the Kaggle competition: Low Cost Motor Imagery Decoding for Rehab (Single Subject). The goal is to classify whether a subject is at rest or attempting to move using EEG recordings collected from a low-cost headset during a rehabilitation-style motor task.

### Why this is challenging

EEG signals are:

- noisy and non-stationary
- subject-specific
- affected by power-line interference and motion artifacts
- highly dependent on preprocessing and signal representation

The solution can be built with either:

- classical signal-processing pipelines combined with machine learning classifiers
- end-to-end deep learning models trained on raw or minimally filtered EEG data

## Pipeline

```text
Raw EEG
  -> channel and trial inspection
  -> filtering and artifact removal
  -> feature extraction or time-domain preparation
  -> model training and validation
  -> final prediction and submission export
```

## Objectives

- inspect EEG structure and recording format
- understand channel and trial layout
- visualize raw and filtered signals
- estimate sampling rate and frequency content
- build a rest-vs-movement classification pipeline
- generate competition-ready predictions

## Workflow

1. Load the EEG data and inspect dimensions
2. Explore channel layout and trial structure
3. Plot raw EEG waveforms
4. Analyze frequency content and interference
5. Apply notch and bandpass filtering when needed
6. Extract features or prepare deep-learning inputs
7. Train and validate the model on held-out trials
8. Generate the final submission file

## Setup

```bash
pip install numpy pandas matplotlib scipy seaborn scikit-learn
```

For deep learning experiments:

```bash
pip install tensorflow torch
```

## Usage

Use the notebooks in this workspace to explore the EEG data, build preprocessing steps, and validate model performance before generating a competition submission.

## License

This project is intended for educational and research use in the context of the Kaggle competition. Please respect the competition rules, dataset terms, and any licensing or usage restrictions.
