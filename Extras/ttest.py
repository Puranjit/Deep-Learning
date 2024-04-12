# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:29:24 2024

@author: puran
"""

import numpy as np
import pandas as  pd
from scipy import stats

# Sample data: Replace these with your actual data
# Assume these are two related datasets (e.g., measurements before and after an intervention)
df = pd.read_excel("TTest.xlsx")
x = df['Yield']


# Perform the paired sample t-test
t_stat, p_value = stats.ttest_rel(df['Yield'], df['Pred'])

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("Reject the null hypothesis - there is a significant difference between the two related groups")
else:
    print("Do not reject the null hypothesis - there is no significant difference between the two related groups")
    
   
PB = pd.Series([1087.32,542.39,544.22, 560.66, 729.77, 1276.69, 1275.51, 599.09, 537.35, 535.78, 645.04, 645.04])
PR = pd.Series([613.01,
634.03,
661.67,
998.08,
880.59,
839.77,
1296.64,
1327.67,
597.12])

A = pd.Series([519.11,
346.96,
335.12,
370.77,
345.74,
])

Fart = pd.Series([1358.69,
1380.94,
1358.69,
1358.69,
1380.94,
1358.69
])

Cielo = pd.Series([692.302,
579.163,
825.994,
601.366,
400.719,
508.982
])

Mead = pd.Series([944.95,
925.861,
915.268,
963.575,
979.58,
927.354
])

Spring = pd.Series([1358.7,
818.5,
707.714,
1375.86,
1147.1,
845.167
])

Jewel = pd.Series([1310.04,
1123.45,
1429.03,
1656.76,
1223.54,
1244.9
])

Star = [556.376,
537.987,
512.463,
532.396,
741.586,
437.521,
590.633,
596.995,
828.502
]

# Hypothesized mean
mu_0 = 661.44 # Replace with your hypothesized mean

# Perform the t-test
t_statistic, p_value = stats.ttest_1samp(Star, mu_0)

# print("t-statistic:", t_statistic)
print("p-value:", p_value)
    
    
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Actual = df['Yield']
Pred = df['Pred']

# Assuming `y_test` is your actual values and `predictions` are the predictions from your model
residuals = Actual - Pred

# Using matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(Pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Using seaborn, which provides a more concise way to create the plot
plt.figure(figsize=(10, 6))
sns.residplot(x=Pred, y=residuals, lowess=True, color="g")
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the differences
    print(md)
    sd = np.std(diff, axis=0)  # Standard deviation of the differences
    print(sd)
    
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='red', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='red')
    plt.axhline(md - 1.96 * sd, color='red')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean Value')
    plt.ylabel('Difference')
    plt.show()
    
bland_altman_plot(Pred, Actual)