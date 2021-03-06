<p align="center">
  <img src="images/tensornet.png" alt="tensornet" />
  <br />
  <br />
  <img src="https://img.shields.io/badge/version-0.9.1-blue.svg" alt="Version">
  <a href="https://github.com/shan18/TensorNet/blob/master/LICENSE"><img src="https://img.shields.io/apm/l/atomic-design-ui.svg?" alt="MIT License"></a>
  <br />
</p>

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
    - Binary Cross Entropy Loss
    - Mean Square Error Loss
    - SSIM and MS-SSIM Loss
    - Dice Loss
  - Evaluation Metrics
    - Accuracy
    - RMSE
    - MAE
    - ABS_REL
  - Optimizers
    - Stochastic Gradient Descent
  - Regularizers
    - L1 regularization
    - L2 regularization
  - LR Schedulers
    - Step LR
    - Reduce LR on Plateau
    - One Cycle Policy
  - LR Range Test
  - Model Checkpointing
  - Tensorboard
- Model training and validation
- Datasets (data is is returned via data loaders)
  - MNIST
  - CIFAR10
  - TinyImageNet
  - MODEST Museum Dataset
- Data Augmentation
  - Resize
  - Padding
  - Random Crop
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
| Learner, Callbacks and Tensorboard | engine |
| Dataset downloading and preprocessing | data |
| GradCAM and GradCAM++ | gradcam |
| Models, loss functions and optimizers | models |
| CUDA setup and result analysis | utils |

For a demo on how to use these modules, refer to the notebooks present in the [examples](https://github.com/shan18/TensorNet/tree/master/examples) directory.

## Contact/Getting Help

If you need any help or want to report a bug, raise an issue in the repo.
