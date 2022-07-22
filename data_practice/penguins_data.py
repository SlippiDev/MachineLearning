from turtle import shape
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

data = pd.read_csv(r'C:\Users\ashan\OneDrive\Documents\GitHub\Practices\Machine_Learning\seaborn-data-master\penguins.csv')
"""
print(data)
print("-----------------------------------------------")
print(data.head(50))
print("-----------------------------------------------")
print(data.describe())
print(data.info())
"""

print(data.groupby("island").size())

# Code
for j in data.index:
    if data.loc[j, 'island'] == 'Biscoe':
        data.loc[j, 'island'] = 1
    elif data.loc[j, 'island'] == 'Dream':
        data.loc[j, 'island'] = 2
    elif data.loc[j, 'island'] == 'Torgersen':
        data.loc[j, 'island'] = 3
print("-------------------------")
for i in data.columns:
    data[i] = np.nan_to_num(data[i])

# print(data)


# data prep
array1 = data.values
X = array1[:, 1:6]
print(X)
Y = array1[:, 0]
print(Y)

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
print(model)
acc = []
from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(X, Y, test_size=0.20, random_state=1)

for i in model:
    results = ms.cross_val_score(i, X_tr, Y_tr, cv = 10, scoring = 'accuracy' )
    sum = 0
    for j in results:
        sum = sum + j
    avg = sum/len(results)
    print("The mean of 10 outcomes of accuracy result of" , i  , "is" , avg , "!")
    acc.append(avg)
print(acc)
d1 = {model[k]: acc[k] for k in range(len(model)) }
print(d1)
for val in d1.items():
    (model1, accc) = val
    if accc == max(d1.values()):
        print("The highest accurate model is {}".format(model1))
        break
plt.bar(['SVC','LR','LD', 'NC', 'DT', 'GB'], acc, data = acc) 
plt.yticks([acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]])
plt.show()
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



