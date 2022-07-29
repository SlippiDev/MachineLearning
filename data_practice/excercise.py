from turtle import shape
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
line = '----------------------------------------------'
data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\MachineLearning\seaborn-data-master\exercise.csv')
#print(data)
# print(data.head(50))
print("-----------------------------------------------")
#print(data.describe())

#print(data.head(100))

scatter_matrix(data, diagonal='hist')
scatter_matrix(data, diagonal='kde')
plt.show()

cormatrix = data.corr()
plt.subplots = data.corr()
sns.heatmap(cormatrix, annot = True)
plt.show()
"""
for j in data.index:
    if data.loc[j, 'kind'] == 'rest':
        data.loc[j, 'kind'] = 1
    elif data.loc[j, 'kind'] == 'walking':
        data.loc[j, 'kind'] = 2
    elif data.loc[j, 'kind'] == 'running':
        data.loc[j, 'kind'] = 3
"""
for j in data.index:
    if data.loc[j, 'diet'] == 'low fat':
        data.loc[j, 'diet'] = 1
    elif data.loc[j, 'diet'] == 'no fat':
        data.loc[j, 'diet'] = 0
print("-------------------------")
for k in data.index:
    if data.loc[k, 'time'] == '1 min':
        data.loc[k, 'time'] = 1
    elif data.loc[k, 'time'] == '15 min':
        data.loc[k, 'time'] = 15
    elif data.loc[k, 'time'] == '30 min':
        data.loc[k, 'time'] = 30


array1 = data.values
# print(array1)

X = array1[:, 1:5]
# print(X)
Y = array1[:, 5]

print(data.info())


print(X)
print(Y)

from sklearn.model_selection import train_test_split as tts

X_tr, X_tst, Y_tr, Y_tst = tts(X, Y, test_size=0.30, random_state=1)
print(line)
print(X_tr, Y_tr, X_tst, Y_tst)

"""
from sklearn.model_selection  import KFold
kf = KFold(n_splits=10, shuffle = True)
for train_index, test_index in kf.split(X):
    X_tr, X_tst = X[train_index], X[test_index]
    Y_tr, Y_tst = Y[train_index], Y[test_index]
"""
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
