# TensorNet

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/shan18/TensorNet/blob/master/LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.7-blue.svg)]()

TensorNet is a high-level deep learning library built on top of PyTorch.

NOTE: This documentation applies to the MASTER version of TensorNet only.

## Installation

You can use pip to install tensornet

```[bash]
pip install torch-tensornet
```

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub

```[bash]
pip install git+https://github.com/shan18/TensorNet.git#egg=torch-tensornet
```

## Features

TensorNet currently supports the following features
- Model architectures
  - ResNet18
  - A custom model called BasicNet
- Model utilities
  - Loss functions
    - Cross Entropy Loss
  - Optimizers
    - Stochastic Gradient Descent
  - Regularizers
    - L1 regularization
    - L2 regularization
  - Callbacks
    - LR Scheduler
  - LR Finder
- Model training and validation
- Datasets (data is is returned via data loaders)
  - CIFAR10
- Data Augmentation
  - Horizontal Flip
  - Vertical Flip
  - Gaussian Blur
  - Random Rotation
  - CutOut
- GradCAM and GradCAM++ (Gradient-weighted Class Activation Map)
- Result Analysis Tools
  - Plotting changes in validation accuracy and loss during model training
  - Displaying correct and incorrect predictions of a trained model


## How to Use

For examples on how to use TensorNet, refer to the [examples](https://github.com/shan18/TensorNet/tree/master/examples) directory.

## Dependencies

TensorNet has the following third-party dependencies
- torch
- torchvision
- torchsummary
- tqdm
- matplotlib
- albumentations
- opencv-python

## Documentation

Documentation making for the library is currently in progress. So until a documentation is available please refer to the following table for various functionalities and their corresponding module names.

| Functionality  | Module Name |
| ------- | ----- |
| Training | train |
| Validation | evaluate |
| Dataset downloading and preprocessing | data |
| GradCAM and GradCAM++ | gradcam |
| Models, loss, optimizers, regularizers and callbacks | model |
| CUDA, random seed and result analysis | utils |

For a demo on how to use these modules, refer to the notebooks present in the [examples](https://github.com/shan18/TensorNet/tree/master/examples) directory.

## Contact/Getting Help

If you need any help or want to report a bug, raise an issue in the repo.
