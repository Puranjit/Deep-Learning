# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 03:23:32 2023

@author: AgCypher: Puranjit Singh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import r2_score

# blueBie = pd.read_excel('Yield pred ML.xlsx') # 89 entries
# blueBie = pd.read_excel('BlueBie 2.xlsx') # 178 entries  - yield same for one tree
# blueBie = pd.read_excel('berryCount.xlsx') # 178 entries  - yield same for one tree
blueBie = pd.read_excel('BlueBook.xlsx') # 178 entries  - yield different for different sides of a tree
# blueBie = pd.read_excel('TTest.xlsx') # 178 entries  - yield same for one tree

grade_mapping = {'13-315': 1, '14-321' : 2,'Cielo' : 3, 'Alapaha' : 4, '14-336' : 5, 'Star' : 6, 'StarB' : 6, '13-291' : 7, 'Powderblue' : 8, 'Premier' : 9, 
                 '14-300' : 10, '14-324' : 11, 'Meadowlark' : 12, 'Springhigh' : 13, '14-323' : 14, '14-296' : 15, 'Farthing' : 16, 'Jewel' : 17, 'A' : 4, 
                 'PR' : 9, 'PB' : 8}

blueBie['Genotype'] = blueBie['Genotype'].map(grade_mapping)

corr = blueBie.corr()

# factors used to develop a LR model
# X = blueBie[['Genotype', 'PlantHeight', 'AvgBerryWt', 'PlantWidth', 'Prediction0.5']]
X = blueBie[['Genotype', 'AvgWt', 'Detection']] 
# X = blueBie[['AvgWt', 'Detection']] 
# X = blueBie[['Detection']] 

# Target variable
y = blueBie['Yield']
# y = blueBie['Berry Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.26, random_state=101, shuffle=False)

clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

from sklearn.linear_model import HuberRegressor
model = HuberRegressor()

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=False)

scores = cross_val_score(lm, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate RMSE (Root Mean Squared Error) for each fold
rmse_scores = np.sqrt(-scores)

print(f"RMSE for each fold: {rmse_scores}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")

# import pingouin as pg
# blueBie = pd.read_excel('ICC test.xlsx') # 178 entries  - yield same for one tree

# data_frame = pd.DataFrame({'actual': blueBie['Yield'], 'predicted': blueBie['Prediction']})

# # Compute ICC(3,1)
# icc_result = pg.intraclass_corr(data=data_frame, targets='ID', raters='Measurement', ratings='Value')
# print(icc_result)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=False)
lm.fit(X_train, y_train)
print(lm.coef_)
print(lm.intercept_)

predictions = lm.predict(X_test)
r2_s = r2_score(y_test, predictions)
sns.scatterplot(x = y_test, y = predictions)
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R2 score is: ', r2_s)

from sklearn.linear_model import HuberRegressor
huber = HuberRegressor()
# Train the model
huber.fit(X_train, y_train)
# Make predictions (for example, on the test set)
predictions = huber.predict(X_test)
r2_s = r2_score(y_test, predictions)
print(r2_s)

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()

# Fit the RANSAC regressor to the data
ransac.fit(X_train, y_train)

# Predict using the trained model
y_pred = ransac.predict(X_test)

r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score is: ', r2_s)

from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GammaRegressor()
model.fit(X_train_scaled, y_train)

# Making predictions (for example, on the test set)
predictions = model.predict(X_test_scaled)

r2_s = r2_score(y_test, predictions)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(r2_s)

from sklearn.linear_model import BayesianRidge

model = BayesianRidge()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
r2_s = r2_score(y_test, predictions)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(r2_s)

from sklearn.neighbors import KNeighborsRegressor
# Create a KNN regressor
regressor = KNeighborsRegressor(n_neighbors=11,weights='uniform', metric='minkowski', leaf_size = 20, p=2)
# regressor = KNeighborsRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score is: ', r2_s)

import lightgbm as lgm

lgm= lgm.LGBMRegressor().fit(X_train, y_train)
y_pred = lgm.predict(X_test)

from sklearn.ensemble import ExtraTreesRegressor

reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
y_pred = reg.predict(X_test)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(reg.score(X_test,y_test))

from sklearn.ensemble import AdaBoostRegressor
reg = AdaBoostRegressor().fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(reg.score(X_test,y_test))

len(y_pred)

k = []
for i in range(len(y_pred)):
    if i%2==0:
        k.append((y_pred[i]+y_pred[i+1])/2)

from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=np.logspace(-1, 2, 10), cv=5)
model.fit(X_train, y_train)

# Making predictions (for example, on the test set)
predictions = model.predict(X_test)        

r2_s = r2_score(y_test, predictions)
print('R2 score is: ', r2_s)

from sklearn.linear_model import ElasticNetCV

model = ElasticNetCV(fit_intercept=False)

# Fit the model to your data
model.fit(X_train, y_train)

# You can also use the model to make predictions
predictions = model.predict(X_test)

r2_s = r2_score(y_test, predictions)
print('R2 score is: ', r2_s)

# And view the coefficients of the model
print("Coefficients:", model.coef_)

from sklearn.svm import SVR

# Create an SVR model
regressor = SVR()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score is: ', r2_s)

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SGDRegressor
# model = SGDRegressor(max_iter=1000, tol=1e-3)
model = SGDRegressor()

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model
r2_s = r2_score(y_test, predictions)
print(f"R2 score is: {r2_s}")

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score is: ', r2_s)

from sklearn.ensemble import RandomForestRegressor
# Create a random forest regressor
regressor = RandomForestRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred2 = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred2)


print('MAE:', metrics.mean_absolute_error(y_test, y_pred2))
print('MSE:', metrics.mean_squared_error(y_test, y_pred2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
print('R2 score is: ', r2_s)

importances = regressor.feature_importances_

# To pair each feature with its importance
feature_importance_dict = dict(zip(X_train.columns, importances))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Create a gradient boosting regressor
regressor = HistGradientBoostingRegressor()
# regressor = GradientBoostingRegressor()
# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_s)

# Save the trained model
model_file = 'trained_model.joblib'  # Choose a file path and name for your model file
joblib.dump(regressor, model_file)

# Testing the exported model
loaded_model = joblib.load(model_file)
# X_test = ...  # Your testing data features

# action = blueBie[['Genotype', 'Berries detected: 0.25', 'Blueberries detected: 0.5']]

# action = blueBie[['Genotype',  'Age', 'Plant height (cm)', 'Plant width (cm)',
#               'Berries detected: 0.25']]
predictions = (loaded_model.predict(X))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, predictions)))

print(predictions)

df = pd.DataFrame({'Ground truth' : y, 'Predictions' : predictions})
df.to_excel('Bluebie2550_v3.xlsx')

# XGBoost
import xgboost as xgb

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Define the parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the model
model = xgb.train(params, dtrain)

# Make predictions on the test set
y_pred = (model.predict(dtest))
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score is: ', r2_s)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2_s = r2_score(y_test, y_pred)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

importances = model.feature_importances_

# To pair each feature with its importance
feature_importance_dict = dict(zip(X_train.columns, importances))


model_file = 'trained_model.joblib'  # Choose a file path and name for your model file
joblib.dump(model, model_file)

# Testing the exported model
loaded_model = joblib.load(model_file)
# X_test = ...  # Your testing data features

X = blueBie[['Genotype', 'PlantHeight', 'AvgBerryWt', 'PlantWidth', 'Prediction0.4']]

dtrain2 = xgb.DMatrix(X, label=y)

# action = blueBie[[             'Berries detected: 0.25']]

# action = blueBie[['Genotype',  'Age', 'Plant height (cm)', 'Plant width (cm)',
#               'Berries detected: 0.25']]
predictions = (loaded_model.predict(dtrain2))

print(predictions)

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Number of splits (k) for cross-validation
k = 5

# Initialize the KFold cross-validation object
kf = KFold(n_splits=k, shuffle=True, random_state=101)

from sklearn.neighbors import KNeighborsRegressor
# Create a list to store the cross-validation results
rmse_scores = []
x = []
r2 = []
from sklearn.svm import SVR


# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    # print(test_index)
    # Split the data into training and testing sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create a gradient boosting regressor
    regressor = RandomForestRegressor()
    
    # # Train the model
    regressor.fit(X_train, y_train)
    
    # # Make predictions on the test set
    # y_pred = regressor.predict(X_test)
    
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test)

    # # Define the parameters for XGBoost
    # params = {
    #     'objective': 'reg:squarederror',
    #     'eval_metric': 'rmse'
    # }

    # # # Train the model
    # model = xgb.train(params, dtrain)
    
    # y_pred = (model.predict(dtest))
    # r2_s = r2_score(y_test, y_pred)
    
    # lm = LinearRegression()
    # lm.fit(X_train, y_train)

    # predictions = lm.predict(X_test)
    # r2_s = r2_score(y_test, predictions)
    
    # x.append(lm.coef_)
    # lm = LinearRegression()
    # lm.fit(X_train, y_train)
    # predictions = lm.predict(X_test)
    # r2_s = r2_score(y_test, predictions)
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    # regressor = RandomForestRegressor()
    # regressor = DecisionTreeRegressor()


    # Train the model

    # Create an SVR model

    # Create a KNN regressor
    # regressor = KNeighborsRegressor()
    # regressor = SVR()
    # regressor.fit(X_train, y_train)

    # # Make predictions on the test set
    y_pred = regressor.predict(X_test)
    r2_s = r2_score(y_test, y_pred)

    # Make predictions on the test set
    # y_pred = (model.predict(dtest))
    
    # Calculate the Mean Squared Error (MSE) for this fold
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    # Append the MSE to the list of scores
    rmse_scores.append(rmse)
    r2.append(r2_s)

# Calculate the mean and standard deviation of the MSE scores
mean_rmse = np.mean(rmse_scores)
mean_r2 = np.mean(r2)

# z=0
# for i in range(len(x)):
#     z += x[i][0]
# z = z/5
# print(z)
    
# std_rmse = np.std(rmse_scores)

# Print the results
print(f"Mean RMSE across {k}-fold cross-validation: {mean_rmse}")
print(f"Mean of R2: {mean_r2}")


# Neural networks

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = Sequential()

# Add the input layer
model.add(Dense(128, activation='relu', 
                input_shape=(X_train_scaled.shape[1],)))

# Add five hidden layers
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=200, batch_size=4, 
          verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Blueberries detected : 0.5

X = blueBie[['Genotype', 
             'Blueberries detected: 0.5']]
X = blueBie[['Genotype',  'Age', 'Plant height (cm)', 'Plant width (cm)',
              'Blueberries detected: 0.5']]

# Target variable
y = blueBie['Total berry']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.25, random_state=101)

# Linear regression

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.coef_)

predictions = lm.predict(X_test)

sns.scatterplot(x = y_test, y = predictions)
plt.show()

# importing metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Create a KNN regressor
regressor = KNeighborsRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Create an SVR model
regressor = SVR()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.ensemble import RandomForestRegressor
# Create a random forest regressor
regressor = RandomForestRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.ensemble import GradientBoostingRegressor

# Create a gradient boosting regressor
regressor = GradientBoostingRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Save the trained model
model_file = 'trained_model.joblib'  # Choose a file path and name for your model file
joblib.dump(regressor, model_file)

# Testing the exported model
loaded_model = joblib.load(model_file)
# X_test = ...  # Your testing data features

# action = blueBie[['Genotype', 
#               'Berries detected: 0.25']]

action = blueBie[['Genotype',  'Age', 'Plant height (cm)', 'Plant width (cm)',
              'Blueberries detected: 0.5']]
action = blueBie[['Genotype',  'Blueberries detected: 0.5']]
predictions = (loaded_model.predict(action)).astype('int')

print(predictions)

# XGBoost
import xgboost as xgb

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Define the parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the model
model = xgb.train(params, dtrain)

# Make predictions on the test set
y_pred = model.predict(dtest)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = Sequential()

# Add the input layer
model.add(Dense(128, activation='relu', 
                input_shape=(X_train_scaled.shape[1],)))

# Add five hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=200, batch_size=4, 
          verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Blueberries detected : 0.175

X = blueBie[['Genotype', 'Plant height (cm)', 'Plant width (cm)',
             'Berries detected 0.175']]

# Target variable
y = blueBie['Total berry']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.25, random_state=101)

# Linear regression

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.coef_)

predictions = lm.predict(X_test)

sns.scatterplot(x = y_test, y = predictions)
plt.show()

# importing metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Create a KNN regressor
regressor = KNeighborsRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Create an SVR model
regressor = SVR()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.ensemble import RandomForestRegressor
# Create a random forest regressor
regressor = RandomForestRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.ensemble import GradientBoostingRegressor

# Create a gradient boosting regressor
regressor = GradientBoostingRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# XGBoost
import xgboost as xgb

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Define the parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the model
model = xgb.train(params, dtrain)

# Make predictions on the test set
y_pred = model.predict(dtest)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = Sequential()

# Add the input layer
model.add(Dense(128, activation='relu', 
                input_shape=(X_train_scaled.shape[1],)))

# Add five hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=200, batch_size=4, 
          verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
