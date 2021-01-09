# ML & DL Algorithms on MNIST Dataset

## Overview

This is a summary on the MNIST hand-written digit classification task for the PRML course. Algorithms practiced include:

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

To save time, the accuracy results were generated using only 10,000 random training samples among total 60,000 ones, and the validation was based on another subset of 5,000 samples. In spite of that, the whole testing set, including 10,000 testing points, was kept.

Model | Validation Accuracy | Test Accuracy
:----:|:----------------:|:---------:
Naive Bayesian | 70.56% | 72.87%
Decision Tree | 79.36% | 78.26%
Logistic Regression | 83.26% | 84.70%
SVM (Linear) | 88.64% | 89.42%
SVM (Polynomial) | 90.72% | 90.17%
SVM (RBF) | 92.28% | 92.20%
K-Nearest Neighbor | 95.30% | 94.88%
Basic CNN | 97.26% | 97.38%
Incepition CNN | 97.02% | 97.70%
Residual CNN | 98.48% | 98.50%
