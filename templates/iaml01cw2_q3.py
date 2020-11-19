
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
from sklearn.decomposition import PCA
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
    # DO KMEANS LIKE ABOVE TO GET CENTRES
    kmeans = KMeans(n_clusters=22, random_state=1)
    kmeans.fit(Xtrn)
    centres = kmeans.cluster_centers_

    # make dataframe combining Xtrn and Ytrn
    x_trn_df = pd.DataFrame(data=Xtrn)
    y_trn_df = pd.DataFrame(data=Ytrn, columns=['lang'])
    xy_trn_df = pd.concat([x_trn_df, y_trn_df], axis=1)

    # group by class
    grouping = xy_trn_df.groupby(xy_trn_df.lang)
    # get means of each class
    langmeans = np.zeros((22,26))
    for i in range(22):
            langmeans[i] = np.mean(grouping.get_group(0).iloc[:,:-1])

    # do pca(n_c=2) on THE MEANS
    pca = PCA(n_components=2)
    meanspca = pca.fit_transform(langmeans)

# on the two PCs, plot the MEAN VECTORS
fig = plt.figure()
ax = fig.add_axes([0.15, 0.11, 0.8, 0.8])
ax.set_title("Comparison of 2D PCA mean vectors and 22-means cluster centres")

meanscatter = ax.scatter(meanspca[:,0], meanspca[:,1], c='orange', s=100, label='Mean vectors')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# on -this same figure-, also plot the cluster centres (transformed into this PC space....)
centrespca = pca.transform(centres)
centrescatter = ax.scatter(centrespca[:,0], centrespca[:,1], c='skyblue', s=100, label='Cluster centres')

# """format it nicely"""
# show lang info with name or abbrev or number
labels=['Ar', 'Ca', 'Cy', 'De', 'En', 'Es', 'Et', 'Fa', 'Fr', 'Id', 'It', 'Ja', 'Lv', 'Mn', 'Nl', 'Ru', 'Sl', 'Sv', 'Pt', 'Ta', 'Tr', 'Zh']

# centres french and means english are overlapping

# annotate every mean except for english, catalan, italian, turkish
for i, label in enumerate(labels):
    if (i!=4 and i!=1 and i!=10 and i!=20):
        plt.annotate(label, (meanspca[:,0][i], meanspca[:,1][i]), color='maroon', xytext=(-5, -5), textcoords='offset points')
    


# annotate english mean
plt.annotate('En', (meanspca[:,0][4], meanspca[:,1][4]), color='maroon', xytext=(2, -4), textcoords='offset points')
# annotate catalan mean
plt.annotate('Ca', (meanspca[:,0][1], meanspca[:,1][1]), color='maroon', xytext=(-8, 0), textcoords='offset points')
# annotate italian mean
plt.annotate('It', (meanspca[:,0][10], meanspca[:,1][10]), color='maroon', xytext=(0, -6), textcoords='offset points')
# annotate turkish mean
plt.annotate('Tr', (meanspca[:,0][20], meanspca[:,1][20]), color='maroon', xytext=(0, -4), textcoords='offset points')


# annotate every centre except for farsi, japanese
for i, label in enumerate(labels):
    if (i!=7):
        plt.annotate(label, (centrespca[:,0][i], centrespca[:,1][i]), color='navy', xytext=(-5, -5), textcoords='offset points')



# annotate farsi centre
plt.annotate('Fa', (centrespca[:,0][7], centrespca[:,1][7]), color='navy', xytext=(-10, -6), textcoords='offset points')




plt.legend()

plt.show()


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
