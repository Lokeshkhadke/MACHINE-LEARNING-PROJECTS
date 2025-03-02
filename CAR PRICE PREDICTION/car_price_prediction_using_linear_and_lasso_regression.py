# -*- coding: utf-8 -*-
"""car price prediction using linear and lasso regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zLD9J5vPNBzm4zu_y_uDX7vUG_GA6uDL

The project aims to predict the selling price of used cars using machine learning techniques. We'll preprocess the dataset, encode categorical variables, train two models (Linear Regression and Lasso Regression), and evaluate their performance. Additionally, we will use visualizations to compare actual and predicted prices to understand model performance better.
"""

#IMPORTING NECESSARY DEPENDENCIES AND LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

#LOADING THE DATASET
dataset = pd.read_csv('/content/car data.csv')

#CHECKING FIRST AND LAST FIVE ROWS OF DATASET

dataset.head()

dataset.tail()

#NUMBER OF ROWS AND COLUMN
dataset.shape

#INFORMATION ABOUT DATASET
dataset.info()

#CHECKING NULL VALUES
print("\nMissing values in each column:")
dataset.isnull().sum()

#CHECKING THE DISTRIBUTION OF CATEGORICAL DATA
print("\nDistribution of Categorical Data:")
print("Seller_Type:", dataset.Seller_Type.value_counts())
print("Transmission:", dataset.Transmission.value_counts())
print("Fuel_Type:", dataset.Fuel_Type.value_counts())

#ENCODING THE CATEGORICAL VARIABLES
dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

#DISPLAYING THE UPDATED DATASET
print(dataset.head())

# Splitting the data into features (X) and target (Y)
X = dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = dataset['Selling_Price']

print("\nFeature matrix (X):")
print(X.head())
print("\nTarget variable (Y):")
print(Y.head())

#SPLITTING THE DATA INTO TRAINING AND TESTING SET
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Displaying dataset shapes
print("\nShapes of datasets:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

#Function to visualize actual vs predicted prices
def visualize_predictions(actual, predicted, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=actual, y=predicted, alpha=0.6, edgecolor='b')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.show()

"""**MODEL1 : LINEAR REGRESSION**"""

print("\n--- Linear Regression Model ---")
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Evaluating Linear Regression model
train_pred_lr = lin_reg_model.predict(X_train)
test_pred_lr = lin_reg_model.predict(X_test)

train_r2_lr = metrics.r2_score(Y_train, train_pred_lr)
test_r2_lr = metrics.r2_score(Y_test, test_pred_lr)

print("Training R-squared score:", train_r2_lr)
print("Testing R-squared score:", test_r2_lr)

# Visualizing actual vs predicted prices for Linear Regression
visualize_predictions(Y_train, train_pred_lr, "Linear Regression - Training Data")
visualize_predictions(Y_test, test_pred_lr, "Linear Regression - Testing Data")

"""**MODEL 2 : LASSO REGRESSION**"""

print("\n--- Lasso Regression Model ---")
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

# Evaluating Lasso Regression model
train_pred_lasso = lass_reg_model.predict(X_train)
test_pred_lasso = lass_reg_model.predict(X_test)

train_r2_lasso = metrics.r2_score(Y_train, train_pred_lasso)
test_r2_lasso = metrics.r2_score(Y_test, test_pred_lasso)

print("Training R-squared score:", train_r2_lasso)
print("Testing R-squared score:", test_r2_lasso)

# Visualizing actual vs predicted prices for Lasso Regression
visualize_predictions(Y_train, train_pred_lasso, "Lasso Regression - Training Data")
visualize_predictions(Y_test, test_pred_lasso, "Lasso Regression - Testing Data")