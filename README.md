# Wine Quality Classification

A comparative study of SVM, Kernel Methods, and Logistic Regression for wine quality prediction.

## Project Overview

Implementation of three classification algorithms from scratch:
- Linear Support Vector Machine (SVM)
- Kernelized SVM with RBF kernel
- Logistic Regression

The models classify wine quality (good/poor) based on physicochemical properties.

## Dataset

- Combined red and white wine samples (6,497 instances)
- 11 physicochemical features
- Binary classification: quality ≥ 6 (+1), quality ≤ 5 (-1)

## Key Features

- **From-scratch implementations** (no sklearn for core algorithms)
- **Proper validation**: Stratified 5-fold CV, train/test separation
- **Automatic hyperparameter tuning**: Random search with CV
- **Class balancing**: Sample weights to handle imbalance (4113 +1, 2384 -1)
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score
