from turtle import shape
from unittest import result
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
line = '----------------------------------------------'
# Data Reading
data = pd.read_csv(r'C:\Users\ashan\OneDrive\Documents\GitHub\Practices\Machine_Learning\seaborn-data-master\flights.csv')

print(data)

print(line)

print(data)
print(line)
print(data.head(50))
print(line)
print(data.describe())
print(line)
for j in data.index:
    if data.loc[j, 'month'] == 'January':
        data.loc[j, 'month'] = 1
    elif data.loc[j, 'month'] == 'February':
        data.loc[j, 'month'] = 2
    elif data.loc[j, 'month'] == 'March':
        data.loc[j, 'month'] = 3
    elif data.loc[j, 'month'] == 'April':
        data.loc[j, 'month'] = 4
    elif data.loc[j, 'month'] == 'May':
        data.loc[j, 'month'] = 5
    elif data.loc[j, 'month'] == 'June':
        data.loc[j, 'month'] = 6
    elif data.loc[j, 'month'] == 'July':
        data.loc[j, 'month'] = 7
    elif data.loc[j, 'month'] == 'August':
        data.loc[j, 'month'] = 8
    elif data.loc[j, 'month'] == 'September':
        data.loc[j, 'month'] = 9
    elif data.loc[j, 'month'] == 'October':
        data.loc[j, 'month'] = 10
    elif data.loc[j, 'month'] == 'November':
        data.loc[j, 'month'] = 11
    elif data.loc[j, 'month'] == 'December':
        data.loc[j, 'month'] = 12
print("-------------------------")

cormatrix = data.corr()
plt.subplots = data.corr()
sns.heatmap(cormatrix, annot = True)
plt.show()
data.plot(kind = 'kde', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()
scatter_matrix(data, diagonal='hist')
scatter_matrix(data, diagonal='kde')
plt.show()


array = data.values
print(array)

X = array[:, 0:2]
print(line)
print(X)
Y =  array[:, 2]
# Y = Y.astype('str')
print(line)
print(Y)
from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(X, Y, test_size=0.20, random_state=1)
print(line)
print(X_tr, Y_tr, X_tst, Y_tst)

# using different models this time for number input/output

from sklearn import datasets, linear_model

model1 = linear_model.Lasso()
import sklearn.model_selection as ms
result = ms.cross_val_score(model1, X_tr, Y_tr, cv = 2, scoring='neg_mean_absolute_error') # this is for numbers

print(result)
print(line)
model1.fit(X_tr, Y_tr)
predictions = model1.predict(X_tst)
print('Prediction is {}'.format(predictions))
print("Cross Check Y Test {}".format(Y_tst))
count = 0
for i in range(len(predictions)):
    if predictions[i] == Y_tst[i]:
        count = count + 1
accuracy = count/len(predictions)
count = 0
for i in range(len(predictions)):
    if predictions[i] == Y_tst[i]:
        count = count + 1
    else: 
        pererror = (predictions[i] - Y_tst[i]) / Y_tst[i] * 100
        # acc1 = 100 - pererror
        count1 = 0
        count1 += pererror
        
        print(pererror)        
accuracy1 = count1/len(predictions)
print(f"The model {model1} had {count} predictions correct, out of {len(predictions)}. The accuracy is {accuracy}!")
print("A low accuracy may mean that the predicted numbers were close to the actual numbers, but not exact.")
print("Here is the close accuracy score: {0}".format(accuracy1))




plt.scatter(Y_tst, predictions)
plt.xlabel('actual')
plt.ylabel('predicted')
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, 'k--')
plt.show()

