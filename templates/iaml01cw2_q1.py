
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
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from iaml01cw2_helpers import *
# from iaml01cw2_my_helpers import *

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
    print(Xtrn_nm[0,:][0:4])
#iaml01cw2_q1_1()   # comment this out when you run the function
print()
print()


      \texttt{Xtrn\_nm[0,:][0:4]}
      
      [-3.13725490e-06, -2.26797386e-05 -1.17973856e-04, -4.07058824e-04]
    
    \vspace{8pt}
    
      \texttt{Xtrn\_nm[-1,:][0:4]}
      
      [-3.13725490e-06, -2.26797386e-05, -1.17973856e-04, -4.07058824e-04]




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


    matplotlib.rcParams.update({'font.size': 22})
    
    f.subplots_adjust(top=0.98, bottom=0.04, left=0.131, right=0.5, hspace=0.1, wspace=0.0)
    plt.show()


#iaml01cw2_q1_2()   # comment this out when you run the function
print()
print()


    \begin{center}
        \includegraphics[scale=0.4]{q1-2.png}
    \end{center}
    As expected, the samples closest to the mean are visually very similar to the mean, and the samples furthest from the mean are visually dissimilar to the mean.
    Anything more???

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q1.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Q1.3")
print("~~~~~~~~~")
def iaml01cw2_q1_3():
    pca = PCA()
    Xtrn_nm_pca = pca.fit_transform(Xtrn_nm)

print()

        \begin{center}[]
        \begin{tabular}{@{}ll@{}}
        \toprule
        Principal Component    & Variance \\ \midrule
        PC1 & 19.81    \\
        PC2 & 12.11    \\
        PC3 & 4.11     \\
        PC4 & 3.38     \\
        PC5 & 2.62     \\ \bottomrule
        \end{tabular}
        \end{center}

# iaml01cw2_q1_3()   # comment this out when you run the function






