
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml01cw2_q1.py', which is the file name you should use when you submit your code for this question.
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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from iaml01cw2_helpers import *

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.1")
print("~~~~~~~~~")
def iaml01cw2_q1_1():
    print(Xtrn_nm[0,:][0:4])
    print()
    print(Xtrn_nm[-1,:][0:4])
#iaml01cw2_q1_1()   # comment this out when you run the function
print()
print()







# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.2")
print("~~~~~~~~~")
def iaml01cw2_q1_2():

    # split instances by class
    byclass = np.zeros((10, 6000, 784))
    for i in range(10):
        byclass[i] = Xtrn[np.where(Ytrn==i)]

    # calculate means for each classs
    classmeans = np.zeros((10, 784))
    for i in range(10):
        classmeans[i] = np.mean(byclass[i], axis=0)

    # calculate euclidean distance between each class's mean and the instances in that class
    dists = np.zeros((10, 6000))
    for i in range(10):         # over classes
        for j in range(6000):   # over instances in each class
            dists[i][j] = np.linalg.norm(classmeans[i] - byclass[i][j])

    # argsort to find which instances are closest and furthest from class means
    inds = np.zeros((10, 6000))
    for i in range(10):
        inds[i] = np.argsort(dists[i])

    # build grid 
    grid = np.zeros((10, 5, 784))
    for i in range(10):
        # closest two
        tmpinds = inds[i][:2]
        tmpinds = tuple(tmpinds.astype(int))
        grid[i][:2] = byclass[i,tmpinds,:]

        # class mean
        grid[i][2] = classmeans[i]

        #furthest two
        tmpinds = inds[i][-2:]
        tmpinds = tuple(tmpinds.astype(int))
        grid[i][-2:] = byclass[i,tmpinds,:]
    




    # plot grid
    f, axarr = plt.subplots(10, 5)
    for i in range(10):
        for j in range(5):

            if (j != 2):
                for k in range(60000):
                    if (Xtrn[k] == grid[i,j]).all():
                        idx = k

            #plt.subplot(10, 5, (i)*5+(j+1))
            img = grid[i,j].reshape((28,28))
            axarr[i,j].imshow(img, cmap='gray_r')
            if (j != 2):
                axarr[i,j].set_title("Class " + str(i) + ",\n sample " + str(idx))
            elif (j == 2):
                axarr[i,j].set_title("Class " + str(i) + " mean")
            axarr[i,j].axis('off')


    #matplotlib.rcParams.update({'font.size': 22})

    plt.show()


#iaml01cw2_q1_2()   # comment this out when you run the function
print()
print()





# Q1.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.3")
print("~~~~~~~~~")
def iaml01cw2_q1_3():
    pca = PCA()
    Xtrn_nm_pca = pca.fit_transform(Xtrn_nm)
    print(pca.explained_variance_[:5])

#iaml01cw2_q1_3()   # comment this out when you run the function
print()
print()




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.4")
print("~~~~~~~~~")
def iaml01cw2_q1_4():
    pca = PCA()
    Xtrn_nm_pca = pca.fit_transform(Xtrn_nm)
    var_cum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(var_cum, linewidth=2, color='firebrick')
    plt.title("Cumulative explained variance ratio over number of principal components")
    plt.xlabel("K (number of principal components)")
    plt.ylabel("Cumulative explained variance")
    plt.show()


#iaml01cw2_q1_4()   # comment this out when you run the function
print()
print()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.5")
print("~~~~~~~~~")
def iaml01cw2_q1_5():
    pca = PCA()
    Xtrn_nm_pca = pca.fit_transform(Xtrn_nm)
    firstten = pca.components_[:10]
    f, axarr = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            idx = i*5 + j;
            img = firstten[idx].reshape((28,28))
            axarr[i,j].imshow(img, cmap='gray_r')
            axarr[i,j].set_title("PC " + str(idx+1))
            axarr[i,j].axis('off')

    plt.show()

#iaml01cw2_q1_5()   # comment this out when you run the function
print()
print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.6")
print("~~~~~~~~~")
def iaml01cw2_q1_6():
    Ks = [5, 20, 50, 200]
    
    # split instances by class
    byclass = np.zeros((10, 6000, 784))
    for i in range(10):
        byclass[i] = Xtrn[np.where(Ytrn==i)]

    # calculate means for each classs
    classmeans = np.zeros((10, 784))
    for i in range(10):
        classmeans[i] = np.mean(byclass[i], axis=0)

    # for each class, for each K...
    for i in range(10):
        for K in Ks:
            # make PCA with K components
            pca = PCA(n_components=K)
            # fit the PCA to the whole class
            Xtrn_nm_pca = pca.fit(byclass[i])
            # dim-reduct the first sample in the class
            dimred = pca.transform(byclass[i][0].reshape(1,-1))
            # reconstruct (inverse-transform) this reduced sample
            reconstr = pca.inverse_transform(dimred).reshape((784,))
            # calculate RMSE
            rmse = math.sqrt( mean_squared_error(byclass[i][0], reconstr) )
            # display
            print("Class: ", i, "\tK = ", K, "  \t RMSE: ", rmse, "\n")
        print()

# iaml01cw2_q1_6()   # comment this out when you run the function
print()
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.7
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.7")
print("~~~~~~~~~")
def iaml01cw2_q1_7():
    Ks = [5, 20, 50, 200]
    imgs = np.zeros((40, 784))
    
    # split instances by class
    byclass = np.zeros((10, 6000, 784))
    for i in range(10):
        byclass[i] = Xtrn[np.where(Ytrn==i)]

    # calculate means for each classs
    classmeans = np.zeros((10, 784))
    for i in range(10):
        classmeans[i] = np.mean(byclass[i], axis=0)

    # for each class, for each K...
    for i in range(10):
        for j in range(4):
            # make PCA with K components
            pca = PCA(n_components=Ks[j])
            # fit the PCA to the whole class
            Xtrn_nm_pca = pca.fit(byclass[i])
            # dim-reduct the first sample in the class
            dimred = pca.transform(byclass[i][0].reshape(1,-1))
            # reconstruct (inverse-transform) this reduced sample
            reconstr = pca.inverse_transform(dimred).reshape((784,))
            # add to imgs to plot
            imgs[i*4 + j] = reconstr
            print(str(i*4 + j) + "...")
            
    # plot
    f, axarr = plt.subplots(10,5, sharex=True, sharey=True)
    for i in range(10):
        for j in range(5):
            if(j==0):
                axarr[i,j].text(10,18, "Class " + str(i))
                axarr[i,j].axis('off')
            else:
                img = imgs[i*4 + j-1].reshape((28,28))
                axarr[i,j].imshow(img, cmap='gray_r')
                axarr[i,j].axis('off')
                if (i==0):
                    axarr[i,j].set_title("K=" + str(Ks[j-1]))
            print(str(i*4 + j) + "...")
    plt.show()


#iaml01cw2_q1_7()   # comment this out when you run the function
print()
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.8")
print("~~~~~~~~~")
def iaml01cw2_q1_8():
    pca = PCA(n_components=2)
    Xtrn_nm_pca2 = pca.fit_transform(Xtrn_nm)
    pcdf = pd.DataFrame(data = Xtrn_nm_pca2, columns = ['PC1', 'PC2'])
    ys = pd.DataFrame(data = Ytrn, columns = ['targ'])
    xydf = pd.concat([pcdf, ys], axis=1)

    plt.title("Normalised training data after PCA reduction to 2 dimensions")
    plt.scatter(xydf.PC1, xydf.PC2, c=xydf.targ, s=1, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()




iaml01cw2_q1_8()   # comment this out when you run the function
print()
print()
