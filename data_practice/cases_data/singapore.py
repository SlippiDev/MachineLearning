from turtle import shape
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\Practices\Machine_Learning\seaborn-data-master\cases.csv')


# EUROPE 

print(data)
print(data.head(50))
print("-----------------------------------------------")
print(data.describe())

print(data.columns)
for i in data.columns:
    print(i)
print(data.loc[1])
print(data.groupby("Europe").size())
"""
#data.plot(kind = "hist", subplots = True, layout = (1,1), sharex = False, sharey = False)
#plt.show()
"""
array = data.values
print(array)
l1 = array[1:, 0]
print(l1)
d1 = 0
list1 = []
for i in l1:
    d1 = d1 + 1
    list1.append(d1)
print(list1)
x = np.array(list1)
x = x.reshape(-1,1)
print(x)
z = array[0:1, 1:160]
print(z)
z = list(z)
country = list(z)
print(country)
print(country[0][3])
y = array[1: ,4]
print(y)
y = y.astype(np.float)
print(y)
y = np.nan_to_num(y)
print(y)
plt.plot(x,y)
plt.show()

"""
"""
scatter_matrix(data, diagonal='hist')
scatter_matrix(data, diagonal='kde')
plt.show()

cormatrix = data.corr()
plt.subplots = data.corr()
sns.heatmap(cormatrix, annot = True)
plt.show()
"""
"""

array1 = data.values
print(array1)
print("-------------------------------------------")
X = array1[:, 0:8]
print(X)
print("----------------------------------")
Y = array1[:, 8]
print(Y)

from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(x, y, test_size=0.20, random_state=1)
 

print(X_tr, Y_tr, X_tst, Y_tst)
model = []
from sklearn import datasets, linear_model
model.append(linear_model.Lasso())
import sklearn.model_selection as ms
print(model)
results = ms.cross_val_score(model[0], X_tr, Y_tr, cv = 10, scoring = "neg_mean_absolute_error")
print(results)

model[0].fit(X_tr, Y_tr)
prediction = model[0].predict(X_tst)

print("prediction:", prediction)
print(Y_tst)

plt.scatter(Y_tst, prediction)
plt.xlabel('actual')
plt.ylabel('predicted')
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, 'k--')
plt.show()

