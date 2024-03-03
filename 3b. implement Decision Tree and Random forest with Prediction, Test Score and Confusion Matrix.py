# Import pandas library
import pandas as pd

# Loading dataset
df = pd.read_csv("D:/MSC 3/ML (1)/ML/prac3/diabetes_dataset.csv")
df.head()

# Feature variables
x = df.drop(['Outcome'], axis=1)

# Target variable
y = df.Outcome

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create Decision Tree classifier object
model = DecisionTreeClassifier()

# Train Decision Tree Classifier
model.fit(x_train, y_train)

# Predict the response for the test dataset
y_pred = model.predict(x_test)

# Evaluation using Accuracy score
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

# Evaluation using Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy calculation from Confusion matrix
accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
print("Accuracy from Confusion Matrix:", accuracy * 100)

# Evaluation using Classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Checking prediction value
prediction = model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
print("Prediction for input:", prediction)

# Import modules for Visualizing Decision trees
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualizing Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=x.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()

# Create Decision Tree classifier object with entropy and max_depth
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifier
model.fit(x_train, y_train)

# Predict the response for the test dataset
y_pred = model.predict(x_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

# Better Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=x.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()




                 
                 
                 
