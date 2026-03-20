# multilayer_perceptron

Implementation of a multilayer perceptron (MLP) for binary classification, developed as part of the 42 School curriculum.

## Overview

The goal of this project is to build a neural network from scratch to classify breast tumors as **malignant (M)** or **benign (B)**.

The model is trained on a dataset describing the characteristics of cell nuclei extracted from breast mass samples.

## Dataset

- CSV dataset with 32 columns
- `diagnosis` is the target label (`M` or `B`)
- The remaining columns are numerical features describing the cell nuclei

## Tech Stack

- Python
- NumPy

## Project Structure

The project is divided into three main modules:

- **Split** — splits the original dataset into training and validation sets
- **Train** — trains the multilayer perceptron on the training dataset
- **Predict** — uses the trained model to make predictions on new data

Each module can be executed from the project root:

    python -m multilayer_perceptron.split.split
    python -m multilayer_perceptron.train.train
    python -m multilayer_perceptron.predict.predict <data-file-path>

## Features

- Multilayer neural network implementation from scratch
- Forward propagation
- Backpropagation
- Dataset splitting (training / validation)
- Binary classification on a real-world dataset
- Early stopping

## Results

**96% prediction accuracy** on unseen dataset

## Learning Outcomes

- Neural network fundamentals
- Gradient descent and backpropagation
- Structuring a machine learning pipeline
- Supervised learning on structured data


![graphs](/screenshots/graphs.png)