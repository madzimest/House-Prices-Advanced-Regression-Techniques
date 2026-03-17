#!/usr/bin/env python3
"""
Script to create the DS project structure with folders and placeholder files.
Run this script from the directory where you want the 'DS' folder to appear.
"""

import json
from pathlib import Path

# Base directory
BASE_DIR = Path.cwd() / "DS"

# Directory structure
DIRS = [
    "app",
    "data/raw",
    "data/processed",
    "models/oof",
    "models/trained",
    "notebooks",
    "src",
    "experiments",
]

# Python source files with brief descriptions (used as file content)
SRC_FILES = {
    "config.py": "# Configuration settings, paths, and hyperparameters.\n",
    "data.py": "# Data loading, preprocessing, and dataset classes.\n",
    "features.py": "# Feature engineering and transformation functions.\n",
    "models.py": "# Model definitions (architectures, wrappers).\n",
    "train.py": "# Training loops and model fitting logic.\n",
    "cv.py": "# Cross-validation strategies and utilities.\n",
    "ensemble.py": "# Ensemble methods (stacking, blending, etc.).\n",
    "utils.py": "# Miscellaneous helper functions.\n",
    "inference.py": "# Inference and prediction pipelines.\n",
}

# Notebook files (empty but valid notebooks with one code cell)
NOTEBOOKS = [
    "01_reproduce_paper.ipynb",
    "02_eda_advanced.ipynb",
    "03_feature_engineering.ipynb",
    "04_modeling_baseline.ipynb",
    "05_model_stacking.ipynb",
]

def create_minimal_notebook(path):
    """Create a minimal valid Jupyter notebook at the given path."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": []
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1)

def create_structure():
    """Create all directories and placeholder files."""
    print(f"Creating project structure in: {BASE_DIR}")

    # Create directories
    for subdir in DIRS:
        dir_path = BASE_DIR / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create Python source files
    for filename, content in SRC_FILES.items():
        file_path = BASE_DIR / "src" / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists, skipping: {file_path}")

    # Create Jupyter notebooks
    for nb in NOTEBOOKS:
        nb_path = BASE_DIR / "notebooks" / nb
        if not nb_path.exists():
            create_minimal_notebook(nb_path)
            print(f"Created notebook: {nb_path}")
        else:
            print(f"Notebook already exists, skipping: {nb_path}")

    # Create requirements.txt (empty, with a comment)
    req_path = BASE_DIR / "requirements.txt"
    if not req_path.exists():
        with open(req_path, "w", encoding="utf-8") as f:
            f.write("# Add project dependencies here\n")
        print(f"Created file: {req_path}")
    else:
        print(f"File already exists, skipping: {req_path}")

    print("\nProject structure created successfully.")

if __name__ == "__main__":
    create_structure()
