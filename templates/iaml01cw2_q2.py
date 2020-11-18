
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml01cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from iaml01cw2_helpers import *
from iaml01cw2_my_helpers import *

datapath = "../data/"
Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(datapath)

Xtrn_orig = Xtrn.copy()
Xtst_orig = Xtst.copy()
twoff_trn = np.ones(Xtrn.shape) * 255.0
twoff_tst = np.ones(Xtst.shape) * 255.0
Xtrn /= twoff_trn
Xtst /= twoff_tst

Xmean = np.mean(Xtrn, axis=0)
Xtrn_nm = Xtrn - Xmean
Xtst_nm = Xtst - Xmean

#<----

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.1")
print("~~~~~~~~~")
def iaml01cw2_q2_1():
    logreg = LogisticRegression() #verbose=1
    logreg.fit(Xtrn_nm, Ytrn)
    preds = logreg.predict(Xtst_nm)
    acc = logreg.score(Xtst_nm, Ytst)
    print("Accuracy: ", acc)
    confusion = confusion_matrix(Ytst, preds)
    print(confusion)
    """
    # plot confusion matrix
    classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', ]
    plt.figure()
    plot_confusion_matrix(confusion, classes=classes, title="Confusion matrix for multinomial logistic regression")
    plt.show()
    """
#iaml01cw2_q2_1()   # comment this out when you run the function
print()
print()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.2")
print("~~~~~~~~~")
def iaml01cw2_q2_2():
    svm = SVC() #verbose=1
    svm.fit(Xtrn_nm, Ytrn)
    preds = svm.predict(Xtst_nm)
    acc = svm.score(Xtst_nm, Ytst)
    print("Accuracy: ", acc)
    confusion = confusion_matrix(Ytst, preds)
    print(confusion)

#iaml01cw2_q2_2()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.3")
print("~~~~~~~~~")
def iaml01cw2_q2_3():
    # do logistic regression
    logreg = LogisticRegression(verbose=1) #
    logreg.fit(Xtrn_nm, Ytrn)


    # get PCs and SDs
    pca = PCA(n_components=2)
    pca.fit(Xtrn_nm)
    V = pca.components_
    var1 = pca.explained_variance_[0]
    var2 = pca.explained_variance_[1]
    sd1 = math.sqrt(var1)
    sd2 = math.sqrt(var2)

    # make 2D grid
    XX, YY = np.mgrid[-5*sd1:5*sd1:0.1*sd1, -5*sd2:5*sd2:0.1*sd2]
    zgrid = np.c_[XX.ravel(), YY.ravel()]

    # transform grid points into original space
        # V    .shape = (    2, 784)
        # zgrid.shape = (10000,   2)
    xgrid = np.dot(zgrid, V)
    preds = logreg.predict(xgrid).reshape(XX.shape)

    # plot graph
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.11, 0.8, 0.8])
    ax.set_title("Decision regions for logistic regression, projected on the first two PCs")
    cont = ax.contourf(XX, YY, preds, cmap='coolwarm')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    cbar = ax.figure.colorbar(cont, ax=ax)

    ax.set_xticks([-5*sd1, -4*sd1, -3*sd1, -2*sd1, -1*sd1, 0, sd1, 2*sd1, 3*sd1, 4*sd1, 5*sd1])
    ax.set_yticks([-5*sd2, -4*sd2, -3*sd2, -2*sd2, -1*sd2, 0, sd2, 2*sd2, 3*sd2, 4*sd2, 5*sd2])

    ax.set_xticklabels(["$-5\sigma_{1}$", "$-4\sigma_{1}$", "$-3\sigma_{1}$", "$-2\sigma_{1}$", "$-\sigma_{1}$", "0", "$\sigma_{1}$", "$2\sigma_{1}$", "$3\sigma_{1}$", "$4\sigma_{1}$", "$5\sigma_{1}$"])
    ax.set_yticklabels(["$-5\sigma_{2}$", "$-4\sigma_{2}$", "$-3\sigma_{2}$", "$-2\sigma_{2}$", "$-\sigma_{2}$", "0", "$\sigma_{2}$", "$2\sigma_{2}$", "$3\sigma_{2}$", "$4\sigma_{2}$", "$5\sigma_{2}$"])

    plt.show()

#iaml01cw2_q2_3()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.4")
print("~~~~~~~~~")
def iaml01cw2_q2_4():
    # do SVM
    svm = SVC()
    svm.fit(Xtrn_nm, Ytrn)
    
    # get PCs and SDs
    pca = PCA(n_components=2)
    pca.fit(Xtrn_nm)
    V = pca.components_
    var1 = pca.explained_variance_[0]
    var2 = pca.explained_variance_[1]
    sd1 = math.sqrt(var1)
    sd2 = math.sqrt(var2)

    # make 2D grid
    XX, YY = np.mgrid[-5*sd1:5*sd1:0.1*sd1, -5*sd2:5*sd2:0.1*sd2]
    zgrid = np.c_[XX.ravel(), YY.ravel()]

    # transform grid points into original space
    xgrid = np.dot(zgrid, V)
    preds = svm.predict(xgrid).reshape(XX.shape)

    # plot graph
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.11, 0.8, 0.8])
    ax.set_title("Decision regions for SVM classification, projected on the first two PCs")
    cont = ax.contourf(XX, YY, preds, cmap='coolwarm')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    cbar = ax.figure.colorbar(cont, ax=ax)

    ax.set_xticks([-5*sd1, -4*sd1, -3*sd1, -2*sd1, -1*sd1, 0, sd1, 2*sd1, 3*sd1, 4*sd1, 5*sd1])
    ax.set_yticks([-5*sd2, -4*sd2, -3*sd2, -2*sd2, -1*sd2, 0, sd2, 2*sd2, 3*sd2, 4*sd2, 5*sd2])

    ax.set_xticklabels(["$-5\sigma_{1}$", "$-4\sigma_{1}$", "$-3\sigma_{1}$", "$-2\sigma_{1}$", "$-\sigma_{1}$", "0", "$\sigma_{1}$", "$2\sigma_{1}$", "$3\sigma_{1}$", "$4\sigma_{1}$", "$5\sigma_{1}$"])
    ax.set_yticklabels(["$-5\sigma_{2}$", "$-4\sigma_{2}$", "$-3\sigma_{2}$", "$-2\sigma_{2}$", "$-\sigma_{2}$", "0", "$\sigma_{2}$", "$2\sigma_{2}$", "$3\sigma_{2}$", "$4\sigma_{2}$", "$5\sigma_{2}$"])

    plt.show()
    
iaml01cw2_q2_4()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.5")
print("~~~~~~~~~")
def iaml01cw2_q2_5():
    # pick the first 1000 from each class to create Xsmall
    # remembering Ysmall too
    bigdf = pd.DataFrame(data = Xtrn_nm)
    bigys = pd.DataFrame(data = Ytrn, columns=['target'])
    bigxy = pd.concat([bigdf, bigys], axis=1)

    # group by class
    grouping = bigxy.groupby(bigxy.target)

    # get first 1000 of each class and combine classes again
    tempgroup = grouping.get_group(0)
    smallxy = tempgroup.head(1000)

    for i in range(1,10):
        tempgroup = grouping.get_group(i)
        smallxy = pd.concat([smallxy, tempgroup.head(1000)])
        
    # separate Xs and Ys
    Xsmall = smallxy.iloc[:,:-1]
    smally = smallxy.iloc[:,-1:]
    Ysmall = smally.values.ravel()
    


    # OKAY

    # 3fold CV with Xsmall only
    Cs = np.logspace(-2, 3, num=10)
    # 3 per 10
    scores = np.zeros((10,3))


    for i in range(10):
        print(i, "...")
        svm = SVC(kernel='rbf', C=Cs[i], gamma='auto')
        scores[i] = cross_val_score(svm, Xsmall, Ysmall, cv=3)
        print("done")
        
    # get means of scores along rows
    meanscores = np.mean(scores, axis=1)

    # plot score means against Cs, remember log scale
    plt.title("Change in SVM average classification accuracy over penalty parameter C")
    plt.xscale('log')
    plt.plot(Cs, meanscores, linewidth=3, color='deeppink')
    plt.xlabel("C (penalty parameter)")
    plt.ylabel("Average accuracy over 3-way cross-validation")



# iaml01cw2_q2_5()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.6")
print("~~~~~~~~~")
def iaml01cw2_q2_6():
    svm = SVC(C=21.5443469)
    svm.fit(Xtrn_nm, Ytrn)
    acc_train = svm.score(Xtrn_nm, Ytrn) 
    acc_test  = svm.score(Xtst_nm, Ytst)
    print("Training accuracy: ", acc_train)
    print("Testing accuracy: ", acc_test)

# iaml01cw2_q2_6()   # comment this out when you run the function
print()
print()
