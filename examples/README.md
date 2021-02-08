# Using TensorNet

This directory contains examples which demonstrate how to use TensorNet.

_NOTE_: These examples apply to the MASTER version of TensorNet only.

## Basic Use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1giK0imE8v1B8hFG02G-nzwTOm1nn9qwq)

The file [tensornet_basic.ipynb](tensornet_basic.ipynb) demonstrates some of the core functionalities of TensorNet:

- Creating and setting up a dataset
- Augmenting the dataset
- Creating and configuring a model and viewing its summary
- Defining an optimizer and a criterion
- Setting up callbacks
- Training and validating the model
- Displaying plots for viewing the change in accuracy during training

## Advanced Use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o3g-tL4c1QPr4KkbPvGyBXPTYh4SoJ1f)

The file [tensornet_advanced.ipynb](tensornet_advanced.ipynb) demonstrates implementation of some of the advanced concepts in Deep Learning using Tensornet:

- Using One Cycle Policy for training
- Running LR Range Test to find the learning rate boundaries for the One Cycle Policy
- Displaying Gradient-weighted Class Activation Maps (GradCAM)
