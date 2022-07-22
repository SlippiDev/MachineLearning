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
#for i in data.columns:
    #data[i] = np.nan_to_num(data[i])
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
Y = Y.astype('str')
print(line)
print(Y)

from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(X, Y, test_size=0.20, random_state=1)
print(line)
print(X_tr, Y_tr, X_tst, Y_tst)
model = []
from sklearn.svm import SVC
model.append(SVC(gamma = 'auto'))
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
import sklearn.model_selection as ms
print(line)
print(model)
print()
acc = []
for i in model:
    results = ms.cross_val_score(i, X_tr, Y_tr, cv = 2, scoring = 'accuracy' )
    print(results)
    
    sum = 0
    for j in results:
        sum = sum + j
    avg = sum/len(results)
    print("The mean of 10 outcomes of accuracy result of" , i  , "is" , avg , "!")
    print()
    acc.append(avg)
print(line)
print(acc)
plt.bar(['SVC','LR','LD', 'NC', 'DT', 'GB'], acc, data = acc) 
# plt.yticks([acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]])
plt.show()

d1 = {model[k]: acc[k] for k in range(len(model)) }
print(line)
print(d1)

for val in d1.items():
    (model1, accc) = val
    if accc == max(d1.values()):
        print("The highest accurate model is {}".format(model1))
        break




model1.fit(X_tr, Y_tr)
predictions = model1.predict(X_tst)
print('Prediction is {}'.format(predictions))
print("Cross Check Y Test {}".format(Y_tst))
count = 0
for i in range(len(predictions)):
    if predictions[i] == Y_tst[i]:
        count = count + 1
accuracy = count/len(predictions)
print(f"The model {model1} had {count} predictions correct, out of {len(predictions)}. The accuracy is {accuracy}!")
plt.scatter(Y_tst, predictions)
plt.xlabel('actual')
plt.ylabel('predicted')
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, 'k--')
plt.show()

