from turtle import shape

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\Practices\Machine_Learning\seaborn-data-master\weather.csv')
print(data)
print(data.head(50))
print("-----------------------------------------------")
#print(data.describe())
# data.plot(kind = "line", subplots = True, layout = (1,2))
# plt.show()
data = data.drop(["Date.Full", "Station.City", "Station.Code", "Station.Location", "Station.State"], axis = 1)

scatter_matrix(data, diagonal='hist')
scatter_matrix(data, diagonal='kde')
plt.show()

cormatrix = data.corr()
plt.subplots = data.corr()
sns.heatmap(cormatrix, annot = True)
plt.show()

print(data.columns)
for i in data.columns:
    print(i)

print(data.loc[1])
print(data.info())
print(data.columns[6])

# print(data.groupby("Station.City").size())
"""
for j in data.index:
    a = 0
    for i in data.loc[j, "Date.Full"]:
        data.loc[j, "Date.Full"] = a
        a += 1
for j in data.index:
    a = 0
    for i in data.loc[j, "Station.City"]:
        data.loc[j, "Station.City"] = a
        a += 1
for j in data.index:
    a = 0
    for i in data.loc[j, "Station.Code"]:
        data.loc[j, "Station.Code"] = a
        a += 1
for j in data.index:
    a = 0
    for i in data.loc[j, "Station.Location"]:
        data.loc[j, "Station.Location"] = a
        a += 1
for j in data.index:
    a = 0
    for i in data.loc[j, "Station.State"]:
        data.loc[j, "Station.State"] = a
        a += 1
"""
array1 = data.values
print(array1)

X = array1[:, 1:15]
print(X)
Y = array1[:, 0]
#Y = Y.astype(np.str)

from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(X, Y, test_size=0.20, random_state=1)
 
print(X_tr, Y_tr, X_tst, Y_tst)

model = []

#from sklearn.svm import SVC
#model.append(SVC(gamma = 'auto'))
from sklearn import datasets, linear_model
model.append(linear_model.Lasso())
"""
from sklearn.linear_model import LogisticRegression as lr
model.append(lr(solver = 'liblinear', multi_class='ovr'))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
model.append(lda())
from sklearn.neighbors import KNeighborsClassifier as knc
model.append(knc())
from sklearn.tree import DecisionTreeClassifier as dtc
model.append(dtc())
from sklearn.naive_bayes import GaussianNB as gnb
model.append(gnb())
"""
import sklearn.model_selection as ms
print(model)

print("-----------------------------------")
acc = []

for i in model:
    results = ms.cross_val_score(i, X_tr, Y_tr, cv = 10, scoring = "neg_mean_squared_error") # doing loop for each model to find model with highest acc 
    sum = 0
    for j in results:
        sum = sum + j
    avg = sum/len(results)
    print("The mean of 10 outcomes of accuracy result of" , i  , "is" , avg , "!")
    acc.append(avg)
print(acc)
"""
d1 = {model[k]: acc[k] for k in range(len(model)) }
print(d1)
for val in d1.items():
    (model1, accc) = val
    if accc == max(d1.values()):
        print("The highest accurate model is {}".format(model1))
"""
#plt.bar(['SVC','LR','LD', 'NC', 'DT', 'GB'], acc, data = acc) 
#plt.yticks([acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]])
#plt.show()
model1 =  model[0]
model1.fit(X_tr, Y_tr)
predictions = model1.predict(X_tst)
print('Prediction is {}'.format(predictions))
print("Cross Check Y Test {}".format(Y_tst))
plt.scatter(Y_tst, predictions)
plt.xlabel('actual')
plt.ylabel('predicted')
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, 'k--')
plt.show()

