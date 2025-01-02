# Diabetes Prediction with Machine Learning

This repository contains a Jupyter Notebook that implements and compares several machine learning models for diabetes prediction.

## Overview

The notebook performs the following steps:

1.  **Data Loading and Exploration:**
    *   Loads the dataset from a CSV file (`diabetes.csv`).
    *   Displays the first few rows and descriptive statistics.
    *   Checks for missing values and duplicates, handling found duplicates.
    *   Prints data information.
2.  **Data Preprocessing:**
    *   Scales the numerical features (e.g., `Pregnancies`, `Glucose`, `BMI`, etc.) using `StandardScaler`.
3.  **Model Training and Evaluation:**
    *   Splits the data into training and test sets (80/20 split).
    *   Trains and evaluates the following models:
        *   **Logistic Regression:** Using `sklearn.linear_model.LogisticRegression`.
        *   **Support Vector Machine (SVM):** Using `sklearn.svm.SVC` with a linear kernel.
        *   **Random Forest:** Using `sklearn.ensemble.RandomForestClassifier`.
    *   Calculates and prints the accuracy score for each model.
4.  **Model Comparison:**
    *   Prints a comparison of all model accuracies.

## Dataset

The dataset used for this project is `diabetes.csv`, which contains the following features:

*   `Pregnancies`: Number of times pregnant.
*   `Glucose`: Plasma glucose concentration.
*   `BloodPressure`: Diastolic blood pressure (mm Hg).
*   `SkinThickness`: Triceps skin fold thickness (mm).
*   `Insulin`: 2-hour serum insulin (mu U/ml).
*   `BMI`: Body mass index (weight in kg/(height in m)^2).
*   `DiabetesPedigreeFunction`: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history).
*   `Age`: Age of the patient.
*   `Outcome`: Class variable (0 or 1 - whether the patient has diabetes).

## Libraries Used

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `matplotlib.pyplot`: For basic plotting (not used in this specific notebook).
*   `seaborn`: For statistical data visualization (not used in this specific notebook).
*   `sklearn.model_selection`: For splitting data into training and testing sets.
*   `sklearn.preprocessing`: For feature scaling (`StandardScaler`).
*   `sklearn.linear_model`: For Logistic Regression model.
*  `sklearn.svm`: for Support Vector Machine model.
*  `sklearn.ensemble`: for Random Forest model.
*   `sklearn.metrics`: For calculating accuracy scores.

## How to Use

1.  Clone this repository to your local machine.
2.  Ensure you have the required libraries installed. You can install them using pip:

    ```bash
    pip install pandas numpy matplotlib scikit-learn seaborn
    ```
3.  Open and run the `diabietis.ipynb` Jupyter Notebook using Jupyter Notebook or Google Colab.
4.  The output of the notebook will display the accuracy of each trained model.

## Results

The notebook will output a summary of the different model performances. The SVM model typically displays the best result for the given data.

## Considerations

*   This notebook serves as a basic starting point. Additional steps such as hyperparameter tuning, cross-validation, and further feature engineering could improve model performance.
*   The choice of best model can depend on the specific needs and constraints of the application.
