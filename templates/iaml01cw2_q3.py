
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
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy as hc
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
            langmeans[i] = np.mean(grouping.get_group(i).iloc[:,:-1])

    print(langmeans)

    # do pca(n_c=2) on THE MEANS
    pca = PCA(n_components=2)
    meanspca = pca.fit_transform(langmeans)

    # on the two PCs, plot the MEAN VECTORS
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.11, 0.8, 0.8])
    ax.set_title("Comparison of PCA mean vectors and K-means cluster centres")

    meanscatter = ax.scatter(meanspca[:,0], meanspca[:,1], c='orange', s=150, label='Mean vectors')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # on -this same figure-, also plot the cluster centres (transformed into this PC space....)
    centrespca = pca.transform(centres)
    centrescatter = ax.scatter(centrespca[:,0], centrespca[:,1], c='deepskyblue', s=150, label='Cluster centres', alpha=0.2)

    # """format it nicely"""
    # show lang info with name or abbrev or number
    labels=['Ar', 'Ca', 'Cy', 'De', 'En', 'Es', 'Et', 'Fa', 'Fr', 'Id', 'It', 'Ja', 'Lv', 'Mn', 'Nl', 'Ru', 'Sl', 'Sv', 'Pt', 'Ta', 'Tr', 'Zh']

    # centres french and means english are overlapping

    # annotate every mean except for english, catalan, italian, slovenian, turkish
    for i, label in enumerate(labels):
        if (i!=4 and i!=1 and i!=10 and i!=17 and i!=20):
            plt.annotate(label, (meanspca[:,0][i], meanspca[:,1][i]), color='maroon', xytext=(-5, -5), textcoords='offset points')
        

    # HORRIBLE
    # annotate english mean
    plt.annotate('En', (meanspca[:,0][4], meanspca[:,1][4]), color='maroon', xytext=(2, -4), textcoords='offset points')
    # annotate catalan mean
    plt.annotate('Ca', (meanspca[:,0][1], meanspca[:,1][1]), color='maroon', xytext=(-8, 0), textcoords='offset points')
    # annotate italian mean
    plt.annotate('It', (meanspca[:,0][10], meanspca[:,1][10]), color='maroon', xytext=(0, -6), textcoords='offset points')
    # annotate Swedish mean
    plt.annotate('Sv', (meanspca[:,0][17], meanspca[:,1][17]), color='maroon', xytext=(-8, -2), textcoords='offset points')
    # annotate turkish mean
    plt.annotate('Tr', (meanspca[:,0][20], meanspca[:,1][20]), color='maroon', xytext=(0, -4), textcoords='offset points')


    # annotate every centre except for farsi
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
    # make dataframe
    x_trn_df = pd.DataFrame(data=Xtrn)
    y_trn_df = pd.DataFrame(data=Ytrn, columns=['lang'])
    xy_trn_df = pd.concat([x_trn_df, y_trn_df], axis=1)
    # group by class
    grouping = xy_trn_df.groupby(xy_trn_df.lang)
    # get means of each class
    langmeans = np.zeros((22, 26))
    for i in range(22):
        langmeans[i] = np.mean(grouping.get_group(i).iloc[:,:-1])
    

    # hierarchy with wards
    wardshc = hc.ward(langmeans)

    # show dendrogram with orientation='right'
    # labels for languages on each leaf
    labels = ["Arabic", "Catalan", "Welsh", "German", "English", "Spanish", "Estonian", "Persian", "French", "Indonesian", "Italian", "Japanesze", "Latvian", "Mongolian", "Dutch", "Russian", "Slovenian", "Swedish", "Portuguese", "Tamil", "Turkish", "Chinese"]
    hc.dendrogram(wardshc, orientation='right', labels=labels)
    
    plt.title("Dendrogram using Ward linkage")
    plt.xlabel("Distance")
    plt.ylabel("Language")
    plt.show()

# iaml01cw2_q3_3()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.4")
print("~~~~~~~~~")
def iaml01cw2_q3_4():
    print()
    # do kmeans(n_clusters=3, random_state=1) FOR EACH LANG CLASS!
    x_trn_df = pd.DataFrame(data=Xtrn)
    y_trn_df = pd.DataFrame(data=Ytrn, columns=['lang'])
    xy_trn_df = pd.concat([x_trn_df, y_trn_df], axis=1)
    grouping = xy_trn_df.groupby(xy_trn_df.lang)

    langvecs = np.zeros((22,3,26))
    
    for i in range(22):
        kmeans = KMeans(n_clusters=3, random_state=1)
        kmeans.fit(grouping.get_group(i).iloc[:,:-1])
        langvecs[i] = kmeans.cluster_centers_



    # should have 66 vecs in total
    # build (22,3,26) and then reshape((66,26)) !! <3
    langvecs = langvecs.reshape(66,26)

    # do hierarchy with ward
    wardhc = hc.ward(langvecs)

    # do hierarchy with single
    singlehc = hc.single(langvecs)

    # do hierarchy with complete
    completehc = hc.complete(langvecs)

    labels = ["Arabic", "Catalan", "Welsh", "German", "English", "Spanish", "Estonian", "Persian", "French", "Indonesian", "Italian", "Japanese", "Latvian", "Mongolian", "Dutch", "Russian", "Slovenian", "Swedish", "Portuguese", "Tamil", "Turkish", "Chinese"]

    kuuskendkuus = [0] * 66

    for i, label in enumerate(labels):
        kuuskendkuus[i*3] = label
        kuuskendkuus[i*3+1] = label
        kuuskendkuus[i*3+2] = label

    # plot dendrogram for ward
    hc.dendrogram(wardhc, orientation='right', labels=kuuskendkuus)
    plt.title("Dendrogram using Ward linkage")
    plt.xlabel("Distance")
    plt.ylabel("Language")
    plt.show()

    # plot dendrogram for single
    hc.dendrogram(singlehc, orientation='right', labels=kuuskendkuus)
    plt.title("Dendrogram using single linkage")
    plt.xlabel("Distance")
    plt.ylabel("Language")
    plt.show()

    # plot dendrogram for complete
    hc.dendrogram(completehc, orientation='right', labels=kuuskendkuus)
    plt.title("Dendrogram using complete linkage")
    plt.xlabel("Distance")
    plt.ylabel("Language")
    plt.show()

# iaml01cw2_q3_4()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q3.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q3.5")
print("~~~~~~~~~")
def iaml01cw2_q3_5():
    # use lang 0 only
    x_trn_df = pd.DataFrame(data=Xtrn)
    y_trn_df = pd.DataFrame(data=Ytrn, columns=['lang'])
    xy_trn_df = pd.concat([x_trn_df, y_trn_df], axis=1)
    group_trn = xy_trn_df.groupby(xy_trn_df.lang)
    lang0_x_trn = group_trn.get_group(0).iloc[:,:-1]

    x_tst_df = pd.DataFrame(data=Xtst)
    y_tst_df = pd.DataFrame(data=Ytst, columns=['lang'])
    xy_tst_df = pd.concat([x_tst_df, y_tst_df], axis=1)
    group_tst = xy_tst_df.groupby(xy_tst_df.lang)
    lang0_x_tst = group_tst.get_group(0).iloc[:,:-1]

    # make zeros array for results
    results = np.zeros((5,2,2))
    # Ks by models by train/test
    # (5,2,2)
    # diag=0, full=1
    # train=0, test=1

    # loop over Ks
    Ks = [1, 3, 5, 10, 15]
    
    for i in range(5):
        print("iteration ", i)
        K = Ks[i]
        gmm_diag = GaussianMixture(n_components=K, covariance_type='diag')
        gmm_full = GaussianMixture(n_components=K, covariance_type='full')
        gmm_diag.fit(lang0_x_trn)
        print("\tdiag model fit")
        gmm_full.fit(lang0_x_trn)
        print("\tfull model fit")
        # full train
        results[i][1][0] = gmm_full.score(lang0_x_trn)
        # full test
        results[i][1][1] = gmm_full.score(lang0_x_tst)
        # diag train
        results[i][0][0] = gmm_diag.score(lang0_x_trn)
        # diag test
        results[i][0][1] = gmm_diag.score(lang0_x_tst)

        print("\tscores calculated!")


    # plot graph...
    plt.title("GMM scores for varying K values, covariance type and scored data")
    # K along xaxis
    # score along yaxis
    # plot full train
    plt.plot(Ks, results[:,1,0], color='lime', marker='o',  label='full-covariance matrix, training data score')
    # plot full test
    plt.plot(Ks, results[:,1,1], color='cyan', marker='o', label='full-covariance matrix, testing data score')
    # plot diag train
    plt.plot(Ks, results[:,0,0], color='deeppink', marker='^', label='diagonal-covariance matrix, training data score')
    # plot diag test
    plt.plot(Ks, results[:,0,1], color='orange', marker='^', label='diagonal-covariance matrix, testing data score')
    plt.xticks(Ks)

    plt.legend()
    plt.show()




iaml01cw2_q3_5()   # comment this out when you run the function
print()
print()
