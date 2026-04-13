# Classical ML Homeworks

This repository contains my coursework notebooks for an introductory machine learning track. The focus is practical classical ML: basic data work in Python, exploratory analysis, preprocessing, train / test splitting, model training, and evaluation with standard metrics. The folder names are kept in the original course submission format.

## What Is Covered Here

- [`hw01_setup_tools`](hw01_setup_tools/hw01_setup_tools.ipynb): environment check, Jupyter setup, NumPy / Pandas basics, vectorization, and simple plotting
- [`hw02_eda`](hw02_eda/hw02_eda.ipynb): Titanic exploratory data analysis, missing values, categorical handling, feature engineering, and visual inspection
- [`hw03_knn`](hw03_knn/hw03_knn.ipynb): KNN classification on Iris with scaling, train / validation / test split, parameter comparison, and classification metrics
- [`hw04_linear`](hw04_linear/hw04_linear.ipynb): linear regression workflow on Toyota data with cleaning, encoding, scaling, and comparison of `LinearRegression`, `Ridge`, and `Lasso`
- [`hw05_logistic`](hw05_logistic/logreg_practice.ipynb): logistic regression from scratch on the breast cancer dataset and comparison with `scikit-learn`

## Topics / Methods

- EDA, descriptive statistics, duplicates, missing values, correlations, and visualizations
- preprocessing with `fillna`, feature engineering, one-hot encoding, and `StandardScaler`
- train / validation / test splitting
- model evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, RMSE, MAE, and R^2
- classical ML algorithms: KNN, linear regression, ridge, lasso, logistic regression

## Repository Structure

- `hw01_setup_tools/`: setup notebook and course notes
- `hw02_eda/`: Titanic EDA notebook and local dataset files
- `hw03_knn/`: KNN notebook and Iris dataset
- `hw04_linear/`: linear regression notebook and Toyota dataset
- `hw05_logistic/`: logistic regression practice notebook
- `requirements.txt`: lightweight dependency list

## Tools Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Note

This is coursework and practice material, not a collection of polished standalone projects. I kept the repository structure close to the course workflow and used the README as the main overview layer.
