# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the plot size
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input Data   
data = pd.read_csv('C:/Users/kamle/Downloads/ML/prac4/data.csv')

# Extracting the features X and Y
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

# Plotting the data points
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data')
plt.show()

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0

for i in range(len(X)):
    num += (X[i] - X_mean) * (Y[i] - Y_mean)
    den += (X[i] - X_mean) ** 2

m = num / den
c = Y_mean - m * X_mean

print("Slope (m):", m)
print("Intercept (c):", c)

# Making Predictions
Y_pred = m * X + c

# Plotting the regression line
plt.scatter(X, Y, label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()
