# Classical ML Homeworks

This repository contains my coursework notebooks for an introductory machine learning track. The work is centered on practical classical ML in Python: data cleaning, exploratory analysis, preprocessing, feature engineering, train / validation / test splitting, model training, and evaluation with standard regression and classification metrics. The folder names are preserved in the original course submission format.

## What Is Covered Here

- [`hw01_setup_tools`](hw01_setup_tools/hw01_setup_tools.ipynb): environment check, Jupyter setup, NumPy / Pandas basics, vectorization, and simple plotting
- [`hw02_eda`](hw02_eda/hw02_eda.ipynb): Titanic exploratory data analysis with missing values, categorical features, basic feature engineering, and visual inspection
- [`hw03_knn`](hw03_knn/hw03_knn.ipynb): KNN classification on Iris with scaling, train / validation / test split, hyperparameter comparison, and classification metrics
- [`hw04_linear`](hw04_linear/hw04_linear.ipynb): linear regression workflow on Toyota data with cleaning, EDA, encoding, scaling, feature engineering, and comparison of `LinearRegression`, `Ridge`, and `Lasso`
- [`hw05_logistic`](hw05_logistic/logreg_practice.ipynb): logistic regression from scratch on the Breast Cancer Wisconsin dataset with gradient descent, log-loss, threshold tuning, and comparison with `scikit-learn`
- [`hw07_decision_tree`](hw07_decision_tree/decision-tree-empty-cells.ipynb): decision tree classification with overfitting analysis, pre-pruning, post-pruning, grid search for hyperparameters, and handling missing values

## Topics / Methods

- EDA, descriptive statistics, duplicates, missing values, correlations, and visualizations
- preprocessing with `fillna`, one-hot encoding, feature engineering, scaling, and data splitting
- train / validation / test workflows for both regression and classification tasks
- model evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, RMSE, MAE, and R^2
- classical ML algorithms: KNN, linear regression, ridge, lasso, logistic regression, and decision trees
- model regularization and control of overfitting through pruning and hyperparameter search

## Repository Structure

- `hw01_setup_tools/`: setup notebook and course notes
- `hw02_eda/`: Titanic EDA notebook and local dataset files
- `hw03_knn/`: KNN notebook and Iris dataset
- `hw04_linear/`: linear regression notebook and Toyota dataset
- `hw05_logistic/`: logistic regression practice notebook
- `hw07_decision_tree/`: decision tree notebook focused on pruning and missing-value handling
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

This is coursework and practice material, not a collection of polished standalone projects. The repository is intentionally kept close to the course workflow, and the README serves as a compact overview of the notebooks and covered ML topics.
