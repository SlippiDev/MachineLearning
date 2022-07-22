from turtle import shape


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer

data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\Practices\Machine_Learning\seaborn-data-master\diabetes_csv.csv')

print(data)
print(data.head(50))
print("-----------------------------------------------")
print(data.describe())

data.plot(kind = "line", subplots = True, )
plt.show()

scatter_matrix(data, diagonal='hist')
scatter_matrix(data, diagonal='kde')
plt.show()

cormatrix = data.corr()
plt.subplots = data.corr()
sns.heatmap(cormatrix, annot = True)
plt.show()

array1 = data.values
print(array1)
print("-------------------------------------------")
X = array1[:, 0:8]
print(X)
print("----------------------------------")
Y = array1[:, 8]
print(Y)

from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(X, Y, test_size=0.20, random_state=1)
 
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
print(model)
print("-----------------------------------")
acc = []
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
"""
# Result Analysis
set_difference = set(predictions) - set(Y_tst)
list_difference = list(set_difference)    
print(f"The model {model1} got ", len(list_difference), "predictions wrong.")
"""
count = 0
for i in range(len(predictions)):
    if predictions[i] == Y_tst[i]:
        count = count + 1
accuracy = count/len(predictions)
print(f"The model {model1} had {count} predictions correct, out of {len(predictions)}. The accuracy is {accuracy}!")
print(model1)