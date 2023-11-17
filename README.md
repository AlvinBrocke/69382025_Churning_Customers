# Churn Prediction App

This repository contains code for a Churn Prediction App built using machine learning. The app predicts whether a customer is likely to churn or not based on various features. The development and training of the machine learning model were performed using Python and popular libraries such as scikit-learn, TensorFlow, and Streamlit.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation & Cleaning (EDA)](#data-preparation--cleaning-eda)
3. [Feature Selection & Importance](#feature-selection--importance)
4. [Training the Multi-Layer Perceptron](#training-the-multi-layer-perceptron)
5. [GridSearchCV and Cross-Validation](#gridsearchcv-and-cross-validation)
6. [Churn Prediction Web App](#churn-prediction-web-app)

## Introduction

The main goal of this project is to predict customer churn in a telecoms company. The project involves data exploration, feature engineering, model training, and the creation of a web app for predictions.

## Data Preparation & Cleaning (EDA)

The initial step involves loading the dataset, exploring its structure, handling missing values, and transforming categorical variables. This is crucial for preparing the data for machine learning.

## Feature Selection & Importance

Feature importance is determined using a Random Forest Classifier, and features with low importance are filtered out. This step helps improve model efficiency and interpretability.

## Training the Multi-Layer Perceptron

A Multi-Layer Perceptron (MLP) model is implemented using TensorFlow's Keras Functional API. The model is trained on the preprocessed data, and the training history is visualized.

## GridSearchCV and Cross-Validation

GridSearchCV is employed to find the best hyperparameters for the MLP model. Cross-validation is used to evaluate the model's performance and generalization to unseen data.

## Churn Prediction Web App

The project includes a Streamlit web app for making real-time predictions. Users can input customer information, and the app displays the likelihood of churn along with confidence levels.

## Usage

To use the Churn Prediction App, follow these steps:

# Jupyter Notebook Instructions

Follow these instructions to run the Jupyter notebook for predicting customer churn in a telecoms company:

1. **Open the Notebook:**

   - Open the Jupyter notebook in your preferred environment.

2. **Dataset Availability:**

   - Ensure that the required dataset, "CustomerChurn_dataset.csv," is available in the specified file path.
   - If needed, update the script with the correct file path.

3. **Sequential Execution:**

   - Run each cell of the notebook sequentially.
   - This ensures that the dependencies are loaded, and the analysis progresses in the intended order.

4. **Organized Sections:**

   - The notebook is organized into sections, each dedicated to a specific aspect of the analysis.
   - Sections include data preparation, exploratory data analysis (EDA), feature selection, training an MLP model, and GridSearchCV with cross-validation.

5. **Saved Model and Scaler:**
   - The script automatically saves the trained model to a file named "best_churn_model.pkl."
   - The scaler used for preprocessing is saved to a file named "scaler.pkl."

Feel free to explore the details within each section of the notebook. If you encounter any issues, ensure that the dataset is correctly located and that the Python environment has the necessary dependencies installed.

Happy analyzing!
