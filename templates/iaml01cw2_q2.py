
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
from sklearn.svm import SVC
from scipy.special import softmax
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


    # get PCs
    pca = PCA(n_components=2)
    pca.fit(Xtrn_nm)
    V = pca.components_.copy()
    var1 = pca.explained_variance_[0]
    var2 = pca.explained_variance_[1]
    sd1 = math.sqrt(var1)
    sd2 = math.sqrt(var2)

    # make 2D grid
    #zXX, zYY = np.mgrid[-5*var1:5*var1:0.1*var1, -5*var2:5*var2:0.1*var2]
    XX, YY = np.mgrid[-5*sd1:5*sd1:0.1*sd1, -5*sd2:5*sd2:0.1*sd2]
    zgrid = np.c_[XX.ravel(), YY.ravel()]

    # transform grid points into original space
        # V    .shape = (    2, 784)
        # zgrid.shape = (10000,   2)
    xgrid = np.dot(zgrid, V)
    probas = logreg.predict_proba(xgrid)[:,1].reshape(XX.shape)

    # plot graph
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.contourf(XX, YY, probas, cmap='coolwarm')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    ax.set_xticks([-5*sd1, -4*sd1, -3*sd1, -2*sd1, -1*sd1, 0, sd1, 2*sd1, 3*sd1, 4*sd1, 5*sd1])
    ax.set_yticks([-5*sd2, -4*sd2, -3*sd2, -2*sd2, -1*sd2, 0, sd2, 2*sd2, 3*sd2, 4*sd2, 5*sd2])

    ax.set_xticklabels(["$-5\sigma_{1}$", "$-4\sigma_{1}$", "$-3\sigma_{1}$", "$-2\sigma_{1}$", "$-\sigma_{1}$", "0", "$\sigma_{1}$", "$2\sigma_{1}$", "$3\sigma_{1}$", "$4\sigma_{1}$", "$5\sigma_{1}$"])
    ax.set_yticklabels(["$-5\sigma_{2}$", "$-4\sigma_{2}$", "$-3\sigma_{2}$", "$-2\sigma_{2}$", "$-\sigma_{2}$", "0", "$\sigma_{2}$", "$2\sigma_{2}$", "$3\sigma_{2}$", "$4\sigma_{2}$", "$5\sigma_{2}$"])




    """
    plt.contourf(XX, YY, probas, cmap='coolwarm')
    plt.colorbar()
    plt.title("VARIANCE!!!!! over something or other whatever look at the graph")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    """
#
iaml01cw2_q2_3()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.4")
print("~~~~~~~~~")
def iaml01cw2_q2_4():
    print()
#
# iaml01cw2_q2_4()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.5")
print("~~~~~~~~~")
def iaml01cw2_q2_5():
    print()
#
# iaml01cw2_q2_5()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q2.6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q2.6")
print("~~~~~~~~~")
def iaml01cw2_q2_6():
    print()
#
# iaml01cw2_q2_6()   # comment this out when you run the function
print()
print()
