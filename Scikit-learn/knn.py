import numpy as np
import pandas as pd
import sklearn
from sklearn import neighbors, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Reading the data
data = pd.read_csv('car.data')
# print(data.head())

# Features (InDependent) - Each feature, or column, represents a measurable piece of data that can be used for analysis
# Labels (Dependent) - The output you get from your model after training it is called a label
# Feature selection - is the process of reducing the number of input variables when developing a predictive model

X = data[['buying', 'maint', 'safety']].values
y = data.loc[:,['class']]
# print(y)
# print(X,y)

# Converting the data
Le = LabelEncoder()

# X - changes in X
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])
# print(X)

# y - changes in y
label_mapping = {
    'unacc' : 0,
    'acc' : 1,
    'good' : 2,
    'vgood' : 3
}

y['class'] = y['class'].map(label_mapping)
y = np.ravel(y)

# Create model
knn = svm.SVC()

# Splitting the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
knn.fit(X_train, y_train)

# Performing prediction
Prediction = knn.predict(X_test)

# Checking the accuracy
accuracy = metrics.accuracy_score(y_test, Prediction)

# print('Prediction: ', Prediction)
# print('Accuracy: ', accuracy)

count = 0
for i in range(len(Prediction)):
    if Prediction[i] == y[i]:
        count+=1
print("Accuracy achieved is: {}".format(count/len(Prediction)*100))

count = 0
for i in range(len(Prediction)):
    if Prediction[i] != y[i]:
        count+=1
print('Loss is: {}'.format(count/len(Prediction)*100))

# Performing predictions
a = 177
print('Actual value: ', y[a])
print('Predicted value: ', knn.predict(X)[a])
