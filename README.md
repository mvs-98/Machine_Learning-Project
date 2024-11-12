# Machine_Learning-Project
Machine Learning project that explores Classification, Regression and Deep Learning methodologies.

```markdown

## Project Overview
This repository contains Classification, Regression and Deep Learning methodologies, which consists of three short tasks covering various machine learning concepts. The project demonstrates key machine learning skills, such as classification, regression, and deep learning.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Dataset](#dataset)
4. [Tasks and Approaches](#tasks-and-approaches)
5. [Results](#results)
6. [Requirements](#requirements)

## Getting Started
Clone this repository to your local machine to get started with the project.

```

## Dataset
The datasets used in this project are provided below and can be downloaded directly:

- [Classification dataset (X)](https://ncl.instructure.com/courses/53509/files/7659751?wrap=1)
- [Classification dataset (y)](https://ncl.instructure.com/courses/53509/files/7659755?wrap=1)
- [Regression dataset](https://ncl.instructure.com/courses/53509/files/7657710?wrap=1)

## Tasks and Approaches
### Task 1: Classification
#### Objective
Build and evaluate two classifiers using different machine learning methods to predict the target class.

#### Approach
1. **Classifier 1 (SVM)**:
   - Implemented a Support Vector Machine (SVM) without hyperparameter tuning and recorded accuracy.
   - Optimized the SVM hyperparameters using `GridSearchCV` and re-evaluated accuracy.
   
2. **Classifier 2 (KNN)**:
   - Built a K-Nearest Neighbors (KNN) classifier initially with `n_neighbors=3`.
   - Conducted hyperparameter tuning with `GridSearchCV` and achieved an optimal model configuration.
   
3. **Evaluation**:
   - Compared models using accuracy, precision, recall, F1-score, and confusion matrix for the best classifier.

### Task 2: Regression
#### Objective
Build and evaluate two regression models to predict target values based on a feature set and optimize the best performing model.

#### Approach
1. **Regression Model 1 (Linear Regression)**:
   - Built a Linear Regression model and evaluated it on the test set.
   
2. **Regression Model 2 (Decision Tree)**:
   - Developed a Decision Tree Regressor with `max_depth=5` and recorded the R² value.
   - Optimized hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) for best performance.

3. **Evaluation**:
   - Reported the R² value and selected the best model.

### Task 3: Deep Learning
#### Objective
Use deep learning models on the MNIST dataset for both an MLP and CNN approach.

#### Approach
1. **MLP Model**:
   - Implemented a Multi-Layer Perceptron (MLP) with three hidden layers of sizes 128, 256, and 50.
   - Trained for two epochs and evaluated accuracy and loss on test data.

2. **CNN Model**:
   - Built a Convolutional Neural Network (CNN) with a single Conv2D layer as the hidden layer.
   - Trained for two epochs, outputted the model structure, and recorded performance metrics.

## Results
- **Classification**: The optimized KNN model achieved the highest accuracy at **92.5%** with detailed precision, recall, F1-score, and confusion matrix metrics.
- **Regression**: The optimized Decision Tree model achieved a high R² value of **0.98**.
- **Deep Learning**: The CNN model achieved **95.87% accuracy** on the MNIST test set with minimal training.

## Requirements
To install the required libraries, use:

```bash
pip install -r requirements.txt
```

### Key Libraries
- **scikit-learn**: For classification and regression algorithms.
- **keras/tensorflow**: For deep learning models on the MNIST dataset.
- **numpy**: For data handling and manipulation.

## How to Run
1. Ensure datasets are in the root directory or specify paths accordingly.
2. Run each question's notebook cells sequentially for results, or use `jupyter nbconvert` for command-line execution.

```
