import tkinter as tk
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import functions




root = tk.Tk()
root.title("Machine Learning")
root.minsize(500,500)

def import_data():
    from tkinter import filedialog as fd
    filename = fd.askopenfilename(initialdir="/", filetypes = ((("Text files", "*.txt*"), ("All files", "*.*"))))
    global data
    data = pd.read_csv(filename)
    print(data)
    print("-------------------------------")
    global array
    array = data.values
    print(array)
    print("--------------------------")


button1 = tk.Button(root, text="Import CSV Data", command = import_data)
button1.grid()
label1 = tk.Label(root, text="File Location")
label1.grid()

def xvalue():
    col = 0
    col = col1.get()
    x = int(col)
    global X
    global y 
    X = array[:, 0:x]
    y = array[:, x]
    print("----------------------------")
    print("Here is X: ")
    print(X)
    print("------------------------")
    print("Here is Y: ")
    print(y)

print("------------------------------")

col1 = tk.Entry(root)
col1.grid()
button3 = tk.Button(root, text = "Select X and Y", command = xvalue)
button3.grid()
def train():   
    from sklearn.model_selection import train_test_split as tts
    global X_tr
    global Y_tr
    global X_tst
    global Y_tst 
    X_tr, X_tst, Y_tr, Y_tst = tts(X, y, test_size=0.20, random_state=1)
    print("------------------------------")
    print(X_tr)
    print("------------------------------")
    print(X_tst)
    print("------------------------------")
    print(Y_tr)
    print("------------------------------")
    print(Y_tst)
    print("------------------------------")

button4 = tk.Button(root, text = "Make Test Train Data",  command = train)
button4.grid()
def evaluation():
    global model
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
        global model1
        (model1, accc) = val
        if accc == max(d1.values()):
            print("The highest accurate model is {}".format(model1))
            break


button5 = tk.Button(root, text = "Evaluate Data", command = evaluation)
button5.grid()

def prediction():
    model1.fit(X_tr, Y_tr)
    global predictions
    predictions = model1.predict(X_tst)
    print("------------------------------")
    print('Prediction is {}'.format(predictions))
    print("--------------------------------")
    print("Cross Check Y Test {}".format(Y_tst))


button6 = tk.Button(root, text = "Predict Data", command = prediction)
button6.grid()

def results():
    global count
    count = 0
    print("----------------------------")
    for i in range(len(predictions)):
        if predictions[i] == Y_tst[i]:
            count = count + 1
    accuracy = count/len(predictions)
    print(f"The model {model1} had {count} predictions correct, out of {len(predictions)}. The accuracy is {accuracy} percent!")


button7 = tk.Button(root, text = "Check Results", command = results)
button7.grid()

def visualize():
    
    # Scatter Matric
    cormatrix = data.corr()
    plt.subplots = data.corr()
    sns.heatmap(cormatrix, annot = True)
    plt.show()
    scatter_matrix(data, diagonal='hist')
    scatter_matrix(data, diagonal='kde')

    # Corelation Matrix
    cormatrix = data.corr()
    plt.subplots = data.corr()
    sns.heatmap(cormatrix, annot = True)
    plt.show()

    # KDE Plot
    data.plot(kind = 'kde', subplots = True, layout = (4,2), sharex = False, sharey = False)
    plt.show()
    

    # Aftermath
    
    plt.scatter(Y_tst, predictions)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.plot(x_lim, y_lim, 'k--')
    plt.show()
    plt.plot(X_tst, Y_tst)
    plt.plot(Y_tst, predictions)
    plt.xticks()
    plt.yticks()
    plt.show()

visualizebutton = tk.Button(root, text = "Visualize Data", command = visualize)
visualizebutton.grid()
root.mainloop()
