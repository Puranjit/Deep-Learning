from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

bc = load_breast_cancer()
# print(bc)

X = scale(bc.data)
print(X)
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

prediction = model.predict(X_test)
labels = model.labels_

print('Labels: ', labels)
print('Prediction: ', prediction)
print('Accuracy is: ', accuracy_score(y_test, prediction))
print('Actual: ', y_test)

from sklearn import metrics

# Commented out IPython magic to ensure Python compatibility.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f',
#           % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean'))

bench_k_means(model, "1", X)

# Classification table
print(pd.crosstab(y_train, labels))