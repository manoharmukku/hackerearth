# HackerEarth - ISI Kolkata Stat Wars 2018
# Stock broker's challenge

import numpy as np
import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from pandas import DataFrame

filename = 'TrainData.csv'
names = ['serial', 'time', 'company', 'buy_price', 'buy_quantity', 'sell_price', 'sell_quantity', 'final_price', 'total_quantity']

# Get the input from the csv file
x_input = pandas.read_csv(filename, header=0, usecols=[3,4,5,6], names=names)
y_input = pandas.read_csv(filename, header=0, usecols=[7], names=names)

# Number of samples
n_samples = x_input.shape[0]

# Normalize the data
x_norm = (x_input - x_input.min()) / (x_input.max() - x_input.min())

# Training set
x_train = x_norm.values
y_train = y_input.values.ravel()

# Create SVM object
svm = SVR(C=1.0, epsilon=0.2)

# Train the model using the training sets
svm.fit(x_train, y_train)

# Read the file to be predicted
x_test = pandas.read_csv('TestData.csv', header=0, usecols=[3,4,5,6], names=names)

# Normalize the test data
x_pred = (x_test - x_test.min()) / (x_test.max() - x_test.min())
x_pred = x_pred.values

# Predict the output on the test data
y_pred = svm.predict(x_pred)

# Convert y_pred to pandas dataframe
y_pred = pandas.DataFrame(y_pred, columns=['nLastTradedPrice'])

# Write output to csv file
y_pred.to_csv('output.csv')

# # Training set
# x_train = x_norm[0 : int(n_samples*0.6)].values
# y_train = y_input[0 : int(n_samples*0.6)].values.ravel()

# # Test set
# x_test = x_norm[int(n_samples*0.6) : n_samples].values
# y_test = y_input[int(n_samples*0.6) : n_samples].values.ravel()

# # Create SVM object
# svm = SVR(C=100, epsilon=0.2)

# # Train the model using the training sets
# svm.fit(x_train, y_train)

# # Make predictions using the testing set
# y_pred = svm.predict(x_test)

# # The mean squared error
# print ("Mean squared error: .%.2f" % mean_squared_error(y_test, y_pred))
# # Explained variance score: 1 is perfect prediction
# print ('Variance score %.2f' % r2_score(y_test, y_pred))