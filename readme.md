# ML & DL Algorithms on MNIST Dataset

## Overview

This is a summary on the MNIST hand-written digit classification task for the PRML course. Algorithms implemented or practiced include:

- K-Nearest Neighbor
- Decision Tree
- Bayesian Decision Theory
- Logistic Regression
- Support Vector Machine (with different kernels)
- Convolutional Neural Network
- CNN with Inceptions
- CNN with Residual Blocks

## Usage

``` powershell
$ python main.py train svm out/model.pkl --tag=train
$ python main.py test svm out/model.pkl --tag=test
```

## Hand-Designed Features

## Results

For time-saving, the accuracy results were generated using only 10,000 random training samples among total 60,000 ones, and the validation was based on another subset of 5,000 samples. In spite of that, the whole testing set, including 10,000 testing points, was kept.

Model | Validation Accuracy | Test Accuracy
:----:|:----------------:|:---------:
SVM (Linear) | 88.64% | 89.42%
SVM (Polynomial) | 90.72% | 90.17%
SVM (RBF) | 92.28% | 92.20%
Basic CNN (3-Layer)| 97.26% | 97.38%
Residual Network | 98.48% | 98.50%
