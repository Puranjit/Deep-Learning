# SVM - creates a hyperplane (support vectors) between different classes

import sklearn
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()

# spliting the datset into features and labels
X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
# print(X.shape)
# print(y.shape)

# Spliting the dataset into train and valid sets for training purposes
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 13)
# print(X_train.shape)
# print(X_valid.shape)
# print(y_train.shape)
# print(y_valid.shape)

# Creating and Training our model
from sklearn import svm

model = svm.SVC()
model.fit(X_train, y_train)

# print(model)

# Predictions
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("Predictions: ", predictions)
print("Accuracy: ", acc)

# print all the class names of Predictions made on Input data
for i in range(len(predictions)):
    print(classes[predictions[i]])