ResNet18 Image Classification using PyTorch
This repository contains a PyTorch implementation of the ResNet18 model for image classification. The model is trained and evaluated on a custom dataset, and the trained model is saved for future use. Below is a detailed explanation of the code, its architecture, and how to use it.
Table of Contents
Overview

Architecture

Code Explanation

Usage

Results

Dependencies

License

Overview
This project demonstrates how to use the ResNet18 architecture for image classification tasks. The model is trained on a custom dataset, validated during training, and finally evaluated on a test set. The trained model is saved to Google Drive for future use. The code is written in PyTorch and includes features like data augmentation, GPU support, and progress bars for training and evaluation.
Architecture
ResNet18
ResNet18 is a convolutional neural network (CNN) architecture that is 18 layers deep. It was introduced in the paper Deep Residual Learning for Image Recognition by He et al. The key innovation of ResNet is the use of residual blocks, which allow the network to learn identity mappings, making it easier to train very deep networks.

Key Features of ResNet18:
Residual Blocks: Each residual block contains two convolutional layers with skip connections that bypass these layers. This helps mitigate the vanishing gradient problem.

Global Average Pooling: Instead of fully connected layers at the end, ResNet uses global average pooling to reduce the spatial dimensions to 1x1.

Pretrained Weights: The model can be initialized with pretrained weights from ImageNet, which helps in transfer learning tasks.
