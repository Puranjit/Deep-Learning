import sklearn
import numpy as np
from sklearn import datasets

# Load iris dataset
iris = datasets.load_iris()

# spliting the datset into features and labels
X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

# Spliting the dataset into train and valid sets for training purposes
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 13)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)