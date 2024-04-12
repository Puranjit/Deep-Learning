# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:46:42 2024

@author: puran
"""

# imports
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import r2_score

shap.initjs()

# data = pd.read_excel('Yield pred ML.xlsx') # 89 entries
# blueBie = pd.read_excel('BlueBie 2.xlsx') # 178 entries  - yield same for one tree
blueBie = pd.read_excel('BlueBook.xlsx') # 178 entries  - yield different for different sides of a tree

# plot 1: whole weight
plt.scatter(data["Detection"], data["Yield"])
plt.ylabel("rings", size=20)
plt.xlabel("whole weight", size=20)

grade_mapping = {'13-315': 1, '14-321' : 2,'Cielo' : 3, 'Alapaha' : 4, '14-336' : 5, 'Star' : 6, 'StarB' : 6, '13-291' : 7, 'Powderblue' : 8, 'Premier' : 9, '14-300' : 10, 
'14-324' : 11, 'Meadowlark' : 12, 'Springhigh' : 13, '14-323' : 14, '14-296' : 15, 'Farthing' : 16, 'Jewel' : 17, '14-323' : 18, 'A' : 4, 'PR' : 9, 'PB' : 8}

data['Genotype'] = data['Genotype'].map(grade_mapping)


cont = ['Genotype', 'AvgWt', 'Detection','Yield']

corr_matrix = pd.DataFrame(data[cont], columns=cont).corr()

sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".1g")

X = data[['Genotype', 'AvgWt', 'Detection']] 

# Target variable
y = data['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=101)

# Linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.coef_)

predictions = lm.predict(X_test)
r2_s = r2_score(y_test, predictions)
sns.scatterplot(x = y_test, y = predictions)
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R2 score is: ', r2_s)


from sklearn.neighbors import KNeighborsRegressor
# Create a KNN regressor
regressor = KNeighborsRegressor(n_neighbors=7, weights='uniform', metric='minkowski', leaf_size = 30, p=2)
# regressor = KNeighborsRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred)

# sns.scatterplot(x = y_test, y = y_pred)
# plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score is: ', r2_s)

# train model
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X, y)

# get shap values
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])



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


# get shap values
explainer = shap.Explainer(regressor)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])

shap.plots.force(shap_values[0])

shap.plots.bar(shap_values)


# SHAP correlation plot
corr_matrix = pd.DataFrame(shap_values.values, columns=X.columns).corr()

sns.set(font_scale=1)
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".1g")


from sklearn.ensemble import GradientBoostingRegressor

# Create a gradient boosting regressor
regressor = GradientBoostingRegressor(n_estimators=30, learning_rate=1e-1,
                                      max_depth=3)

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_s)

import xgboost

model_depth3 = xgboost.XGBRegressor(
    learning_rate=0.02,
    subsample=0.2,
    colsample_bytree=0.5,
    n_estimators=5000,
    base_score=y_train.mean(),
)
model_depth3.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="logloss",
    verbose=500,
    early_stopping_rounds=20,
)

y_pred = model_depth3.predict(X_test)
r2_s = r2_score(y_test, y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_s)


shap_values = shap.TreeExplainer(model_depth3).shap_values(X_test)
shap_interaction_values = shap.TreeExplainer(model_depth3).shap_interaction_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")


e3 = shap.TreeExplainer(model_depth3)
shap_values3 = e3.shap_values(X_test)
shap_interaction_values3 = shap.TreeExplainer(model_depth3).shap_interaction_values(X_test)

shap.summary_plot(shap_values3, X_test, plot_type="bar")













import qrcode

# Create a QR code object with a larger size and higher error correction
qr = qrcode.QRCode(version=3, box_size=20, border=10, error_correction=qrcode.constants.ERROR_CORRECT_H)

# Define the data to be encoded in the QR code
data = "https://auburn.box.com/s/9skyb7gdeylq2fszxub7nvqz4h8zrw2v"
# Add the data to the QR code object
qr.add_data(data)

# Make the QR code
qr.make(fit=True)

# Create an image from the QR code with a black fill color and white background
img = qr.make_image(fill_color="black", back_color="white")

# Save the QR code image
img.save("BlueBie App.png")