# DL-Based-IDS

# Intrusion Detection Scripts

This repository contains Python scripts used in our research on intrusion detection for automotive Ethernet AVB networks. Below is a description of the available scripts and their purposes.

## Repository Structure

- `feature_generator.py`: 
  Implements the feature generation process for training datasets. This script includes functions to preprocess raw data, compute deltas, and split byte sequences into nibbles for input into neural networks.

- `2d_cnn_full_model.py`: 
  Defines the architecture of a 2D-CNN model used for binary classification. This model processes the input features generated and outputs predictions for intrusion detection.

- `distillation_models.py`: 
  Implements knowledge distillation techniques to train student models using a pre-trained teacher model. Includes both a standard student model and an ultra-light version for resource-constrained environments.

- `pruning_optimization.py`: 
  Demonstrates neural network pruning using TensorFlow Model Optimization Toolkit (TF-MOT). This script reduces the size of the neural network by pruning unnecessary weights while maintaining accuracy.

## Important Notes

The scripts provided in this repository are the **skeletons** of the code used in our research. They include only the essential logic and structure, designed to illustrate the core methods and techniques. 
