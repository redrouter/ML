import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.title("Scatter Plot of Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

plt.scatter(train_x,train_y)
plt.show()

plt.scatter(test_x,test_y)
plt.show()

degree = 4  
train_model = np.poly1d(np.polyfit(train_x, train_y, degree))
myline = np.linspace(0, 6, 200)

plt.figure(figsize=(8, 6))
plt.scatter(train_x, train_y)
plt.plot(myline, train_model(myline))
plt.title("Polynomial Regression Model (Training Data)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

r2_train = r2_score(train_y, train_model(train_x))
print("R-squared score for training data:", r2_train)

test_model = np.poly1d(np.polyfit(test_x, test_y, degree))

plt.figure(figsize=(8, 6))
plt.scatter(test_x, test_y)
plt.plot(myline, test_model(myline))
plt.title("Polynomial Regression Model (Testing Data)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

r2_test = r2_score(test_y, test_model(test_x))
print("R-squared score for testing data:", r2_test)

prediction = test_model(5)
print("Prediction for x = 5:", prediction)

