# Ames Housing Price Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A complete end‑to‑end machine learning pipeline for predicting house sale prices using the **Ames Housing dataset**. The project implements data preprocessing, feature engineering, cross‑validation, multiple regression models (Ridge, Random Forest, LightGBM, XGBoost), and advanced ensemble techniques (blending and stacking). It achieves a final RMSE below **$21,000** on the test set, well within top‑tier Kaggle performance.

The project is structured as a reusable Python package (`src/`) and includes a **Flask API** for real‑time predictions, making it suitable for both learning and deployment.


## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Feature Engineering](#feature-engineering)
  - [Training Models](#training-models)
  - [Cross‑Validation](#cross‑validation)
  - [Ensemble (Blending & Stacking)](#ensemble-blending--stacking)
  - [Making Predictions](#making-predictions)
  - [Web API](#web-api)
- [Results](#results)
- [Notebooks](#notebooks)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Project Overview

The goal is to predict the final sale price of homes in Ames, Iowa, using 79 explanatory variables describing (almost) every aspect of residential homes. This project is inspired by the famous [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Key highlights:**

- **Modular codebase**: All functionality is organized into `src/` modules (config, data, features, models, cv, ensemble, train, inference, utils).
- **Robust cross‑validation**: Uses k‑fold (default 10) to generate out‑of‑fold predictions and avoid overfitting.
- **Feature engineering**: Creates interaction features, aggregates, ratios, and handles missing values systematically.
- **Multiple models**: Ridge, Random Forest, LightGBM, XGBoost – each with hyperparameters tuned for this dataset.
- **Advanced ensembles**: Implements blending (weighted average) and stacking (meta‑learner) to combine model strengths.
- **Flask API**: Serves predictions via a simple REST endpoint.


## Dataset

The **Ames Housing dataset** was compiled by Dean De Cock and is a modern alternative to the classic Boston Housing dataset. It contains 2,930 observations with 79 features (23 nominal, 23 ordinal, 14 discrete, and 20 continuous). The target variable is `SalePrice`.

- **Training set**: 1,460 samples (after removing two extreme outliers)
- **Test set**: 1,459 samples (without labels)
- **Data files** (after running the setup script):
  - `data/raw/train.csv`
  - `data/raw/test.csv`

The dataset is automatically downloaded when you run the provided scripts (see [Installation](#installation)).


## Project Structure

The project is organized according to standard data science project conventions:
```
├── House-Prices-Advanced-Regression-Techniques
│   ├── all_files.txt
│   ├── app
│   │   └── app.py
│   ├── create_ds_structure.py
│   ├── data
│   │   ├── processed
│   │   └── raw
│   │       ├── data_description.txt
│   │       ├── sample_submission.csv
│   │       ├── test.csv
│   │       └── train.csv
│   ├── deployment
│   │   └── stacking_pipeline.pkl
│   ├── experiments
│   ├── loop_getFile_contents.py
│   ├── models
│   │   ├── oof
│   │   └── trained
│   │       └── blend_run1
│   │           ├── fold_0.pkl
│   │           ├── fold_1.pkl
│   │           ├── fold_2.pkl
│   │           ├── fold_3.pkl
│   │           ├── fold_4.pkl
│   │           ├── fold_5.pkl
│   │           ├── fold_6.pkl
│   │           ├── fold_7.pkl
│   │           ├── fold_8.pkl
│   │           └── fold_9.pkl
│   ├── notebooks
│   │   ├── 01_reproduce_paper.ipynb
│   │   ├── 02_eda_advanced.ipynb
│   │   ├── 03_feature_engineering.ipynb
│   │   ├── 04_modeling_baseline.ipynb
│   │   ├── 05_model_stacking.ipynb
│   │   ├── house-prices-advanced-regression-techniques.ipynb
│   │   └── submission.csv
│   ├── readme.md
│   ├── requirements.txt
│   └── src
│       ├── config.py
│       ├── cv.py
│       ├── data.py
│       ├── ensemble.py
│       ├── features.py
│       ├── inference.py
│       ├── __init__.py
│       ├── models.py
│       ├── __pycache__
│       │   ├── config.cpython-310.pyc
│       │   ├── cv.cpython-310.pyc
│       │   ├── data.cpython-310.pyc
│       │   ├── ensemble.cpython-310.pyc
│       │   ├── features.cpython-310.pyc
│       │   ├── inference.cpython-310.pyc
│       │   ├── __init__.cpython-310.pyc
│       │   ├── models.cpython-310.pyc
│       │   ├── train.cpython-310.pyc
│       │   └── utils.cpython-310.pyc
│       ├── train.py
│       └── utils.py
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ames-housing-prediction.git
cd ames-housing-prediction


python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
