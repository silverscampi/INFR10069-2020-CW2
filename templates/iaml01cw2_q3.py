
##########################################################
#  Python script template for Question 3 (IAML Level 10)
#  Note that:
#  - You should not change the name of this file, 'iaml01cw2_q3.py', which is the file name you should use when you submit your code for this question.
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
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from iaml01cw2_helpers import *
from iaml01cw2_my_helpers import *

#<----

Xtrn, Ytrn, Xtst, Ytst = load_CoVoST2("../data/")

# random_state=1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.1")
print("~~~~~~~~~")
def iaml01cw2_q3_1():
    kmeans = KMeans(n_clusters=22, random_state=1)
    kmeans.fit(Xtrn)
    inertia = kmeans.intertia_

    for i in range(22):
        print("for label ", i, "\t", (kmeans.labels_ == i).sum())


# iaml01cw2_q3_1()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.2")
print("~~~~~~~~~")
def iaml01cw2_q3_2():
    # make dataframe combining Xtrn and Ytrn

    # np.mean(xy_trn_df, axis=0).shape = (26,)


    # do pca(n_c=2) on THE MEANS

    # on the two PCs, plot the MEAN VECTORS


    # on -this same figure-, also plot the cluster centres (transformed into this PC space....)

    # """format it nicely"""
    # show lang info with name or abbrev or number

# iaml01cw2_q3_2()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.3")
print("~~~~~~~~~")
def iaml01cw2_q3_3():
    # get means again as above


    # hierarchy with wards


    # show dendrogram with orientation='right'
    # labels for languages on each leaf
# iaml01cw2_q3_3()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.4")
print("~~~~~~~~~")
def iaml01cw2_q3_4():
    # do kmeans(n_clusters=3, random_state=1) FOR EACH LANG CLASS!

    # do hierarchy with ward

    # do hierarchy with single

    # do hierarchy with complete


    # plot dendrogram for ward

    # plot dendrogram for single

    # plot dendrogram for complete


# iaml01cw2_q3_4()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.5")
print("~~~~~~~~~")
def iaml01cw2_q3_5():
    print()
# iaml01cw2_q3_5()   # comment this out when you run the function
print()
print()
