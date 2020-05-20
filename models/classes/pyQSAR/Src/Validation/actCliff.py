'''
Identify Activity Cliffs

DESCRIPTION
    This module uses the MODI formalism to help determine activity cliffs through clustering.

NOTES
    !!! WORK IN PROGRESS !!!
'''

# Imports
import pandas as pd
import numpy as np
import scipy.spatial.distance as scd
import sklearn.neighbors as skn
import sklearn.cluster as skdb
import matplotlib.pyplot as plt

# Functions
def actCliff_Contribution(inDF):
    '''
    Determine the contribution of each compound to the MODI based on activity cliffs.

    INPUT
        inDF: (pandas Data Frame) Input data frame where the first column is the activity value and the remaining are descriptors.

    OUTPUT
    '''

    # Variables
    actChangeList = []
    indList = []
    bruteForce = []

    # Calculate nearest neighbors
    knn = 2
    tree = skn.KDTree(inDF.iloc[:,1:], leaf_size=100)
    dist, ind = tree.query(inDF.iloc[:,1:], k=knn, sort_results=True)

    # Brute force
    colNameList = inDF.columns.values
    print(len(colNameList))
    for index in range(3):
        distVal = 0.0
        for index2,colName in enumerate(colNameList):
            distVal += (inDF.iloc[index,index2]-inDF.iloc[236,index2])**2

            if (index == 0) and (index2 > len(colNameList)-4):
                print(distVal)

        bruteForce.append(np.sqrt(distVal))

    # Remove self-distance indices
    for index,cmp in enumerate(ind):
        indList.append(np.delete(ind[index],0)[0])

    for index,val in enumerate(zip(dist,ind)):
        if (index < 3):
            print(val)

    # Calculate change in activity
    for index,distance in enumerate(dist):
        actChangeList.append(abs(inDF.iloc[index,0]-inDF.iloc[indList[index],0]))

    print(bruteForce)
    #print(actChangeList)

# Main
if (__name__ == '__main__'):
    pass
