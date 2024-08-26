# kNN-on-the-Breast-Cancer-Dataset
### Project Overview

This project aims to classify breast cancer tumors as benign or malignant using the K-Nearest Neighbors (KNN) algorithm. The dataset used contains various features computed from breast mass imaging, and the target is to determine whether a tumor is malignant (cancerous) or benign (non-cancerous). 

The purpose behind this project is applying the notions studied for the Artificial Intelligence I course to further my understanding and improving my Python skills.

### Dataset

The dataset used in this project is the [Breast Cancer Wisconsin Diagnostic Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), available from the UCI Machine Learning Repository.


### Dataset Features
ID: Identification number.
Diagnosis: Target variable (M = malignant, B = benign).
Features: 30 numeric features representing various properties of cell nuclei (e.g., radius, texture, perimeter, area, smoothness).
- Number of Instances: 569 total entries
- Number of Features: 30 numeric features (excluding the target).
- Target Classes:
   - Malignant (denoted as 1)
   - Benign (denoted as 0)

### Prerequisites
Python 3.x, pip (Python package installer)

### Required Libraries
The project uses the following Python libraries:
- pandas: For data manipulation and analysis.
- numpy: For numerical computations.
- scikit-learn: For machine learning modeling and evaluation.
- matplotlib & seaborn: For data visualization.

### Exploratory Data Analysis (EDA)

The EDA section of the notebook/script provides insights into the dataset, including:

- Distribution of the target variable (Diagnosis).
- Summary statistics of the features.
- Pair plots and correlation heatmaps to identify relationships between features.

### Modeling

K-Nearest Neighbors (KNN)
The KNN algorithm is used to classify tumors as malignant or benign based on the nearest neighbors in the feature space. The model is evaluated using metrics such as accuracy, precision, and recall.

### Data Preprocessing
Scaling: The features are standardized using StandardScaler to ensure that each feature contributes equally to the distance calculations.

Train/Test Split: The data is split into training and testing sets.

### Hyperparameter Tuning
The optimal value of k (number of neighbors) is determined using cross-validation.

### Results

The KNN model achieved an accuracy score of 95.9%.
