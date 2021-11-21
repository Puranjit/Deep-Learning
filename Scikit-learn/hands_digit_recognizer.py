import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

mnist = load_digits()

X = mnist.data
y = mnist.target

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train/256)
X_test = np.array(X_test/256)

print(y_train.shape)

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64))
print(clf)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Predictions: ', predictions)

acc = clf.score(X_test, y_test)
print('accuracy: ', acc)

accu = confusion_matrix(y_test, predictions)
print(accu)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements
print('Accuracy is: {}'.format(accuracy(accu)))