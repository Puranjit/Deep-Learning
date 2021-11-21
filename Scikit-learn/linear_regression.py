# Linear regression
# Import libraries

from sklearn import datasets
boston = datasets.load_boston()

# features/labels
X = boston.data
y = boston.target

# print(X)
# print(X.shape)
# print('\n', y)
# print(y.shape)

from sklearn import linear_model
# Creating model
l_reg = linear_model.LinearRegression()

from matplotlib import pyplot as plt
plt.scatter(X.T[5],y)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training our model
l_reg.fit(X_train, y_train)

# Prediction
predictions = l_reg.predict(X_test)
print("Predictions: ", predictions)

# R^2 : is the proportion of the variation in the dependent variable that is predictable from the independent variable
print("R^2 value: ", l_reg.score(X, y))

# Coeff : Slope for each pt.
print('Coeff.: ', l_reg.coef_)

# Coeff : Intercept for feature 
print('Intercept: ', l_reg.intercept_)