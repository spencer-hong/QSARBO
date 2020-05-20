'''
modelability

DESCRIPTION
  Calculate modelability.
'''

# Imports
import copy
import pandas as pd
import numpy as np
import scipy.spatial.distance as scd
import sklearn.neighbors as skn
import sklearn.cluster as skc
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


# Functions
def threshLine(activity):
    '''
    Line for determining threshold.

    INPUT
        activity: (float) Activity value.

    NOTES
        The threshold for knn activity cliff identification cannot be constant due to the distribution of the activities. Therefore, a higher threshold will be assigned to smaller values than larger values.
    '''

    slope = -2.5/4.0

    return slope*activity+3.0

def plot_dendrogram(model, **kwargs):
    '''
    Plot dendrogram.

    NOTES
        Adopted from https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
    '''

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    sch.dendrogram(linkage_matrix, **kwargs)

def show_hierarchical_clustering(inDF):
    '''
    Determine clustering and plot dendrogram of data set.

    INPUT
        inDF: (pandas Data Frame) Input data frame with activity and features.
    '''

    # Set up clustering model
    model = skc.AgglomerativeClustering(n_clusters=10)

    # Perform clustering
    model = model.fit(inDF.iloc[:,1:])

    # Plot clusters
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, labels=model.labels_)
    plt.show()

def calcNN(data,knn=2):
    '''
    Calculate nearest neighbors using KDTree.

    INPUT
        data: (numpy array of numbers) Numpy array containing activity values.
        knn: (int) Number of nearest-neighbors to find.

    OUTPUT
        outDist: (numpy array of numbers) Numpy array of distances between neighbors.
        outInd: (numpy array of ints) Numpy array of indices of neighbors.
    '''

    # Calculate index of nearest neighbor using SKLearn
    if (knn == 2):
        tree = skn.KDTree(data[:,1:], leaf_size=2)
        dist, ind = tree.query(data[:,1:], k=knn)
    else:
        tree = skn.KDTree(data, leaf_size=2)
        dist, ind = tree.query(data, k=knn)

    # Convert to a list for deleting first elements
    dist = dist.tolist()
    ind = ind.tolist()

    # Remove self-distance indices
    for index,cmp in enumerate(ind):
        ind[index] = np.delete(ind[index],0)
        dist[index] = np.delete(dist[index],0)

    outDist = np.asarray(dist)
    print("Average Distance: " + str(np.mean(outDist)))
    outInd = np.asarray(ind)

    return outDist,outInd

def rMODI(inDF,method='spectral',plot=False):
    '''
    Regression MODI with choice of method to classify each activity value.

    INPUT
        inDF: (pandas Data Frame) Input data frame with only the activity values.

        method: (str) Method to use for classification.

        plot: (boolean) Decide whether to plot resulting groups.

    OUTPUT
        MODIVal: (float) MODI value.

        cliffInd: (list of ints) List of indices of compounds which contribute to cliffs.
    '''

    # Variables
    inDataCpy = copy.deepcopy(inDF)
    cliffInd = []

    # Convert pandas dataframe to numpy array
    data = inDF.iloc[:,1:].values

    # Set up clustering model
    if (method == 'spectral'):
        cluster_model = skc.SpectralClustering(n_clusters=10,
                                               random_state=42,
                                               n_jobs=7)
        classes = cluster_model.fit_predict(data)

    elif (method == 'kmeans'):
        cluster_model = skc.KMeans(n_clusters=10,
                                   random_state=42,
                                   n_jobs=7)
        classes = cluster_model.fit_predict(data)

    elif (method == 'ward'):
        cluster_model = skc.AgglomerativeClustering(n_clusters=10)
        classes = cluster_model.fit_predict(data)

    elif (method == 'dbscan'):
        cluster_model = skc.DBSCAN(eps=0.5,
                                   min_samples=3,
                                   n_jobs=7)
        classes = cluster_model.fit_predict(data)

    elif (method == 'knn'):
        # Variables
        #thresh = 0.99       # Precentage threshold value for cliff identification
        cliffIdx = []       # List for cliff indices

        # Calculate k nearest neighbors
        tree = skn.KDTree(inDataCpy.iloc[:,1:], leaf_size=2)
        dist, ind = tree.query(inDataCpy.iloc[:,1:], k=3)

        # Convert to a list for deleting first elements
        dist = dist.tolist()
        ind = ind.tolist()

        # Remove self-distance indices
        for index,cmp in enumerate(ind):
            ind[index] = np.delete(ind[index],0)
            dist[index] = np.delete(dist[index],0)

        # Convert index list into lists of values
        ind = [list(x) for x in ind]

        # Identify cliffs
        for listIdx,indList in enumerate(ind):
            act1 = inDataCpy.iloc[listIdx,0]
            for idx in indList:
                act2 = inDataCpy.iloc[idx,0]

                # Check activity difference
                percentDiff = abs((act1-act2)/act2)

                if (percentDiff >= threshLine(act1)):
                    if (listIdx not in cliffIdx):
                        cliffIdx.append(listIdx)

                    if (idx not in cliffIdx):
                        cliffIdx.append(listIdx)

        # Calculate MODI
        MODIVal = 1.0*len(cliffIdx)/(inDataCpy.shape)[0]

        return MODIVal,cliffIdx

    # Apply clustering
    inDataCpy.iloc[:,0] = classes

    # Use classification MODI
    MODIVal,cliffInd = cMODI(inDataCpy)

    return MODIVal,cliffInd

def rMODI_Spectral(inDF,plot=False):
    '''
    Regression MODI using spectral clustering to classify each activity value.

    INPUT
        inDF: (pandas Data Frame) Input data frame with only the activity values.
        plot: (boolean) Decide whether to plot resulting groups.

    OUTPUT
        MODIVal: (float) MODI value.

        cliffInd: (list of ints) List of indices of compounds which contribute to cliffs.
    '''

    # Variables
    inDataCpy = copy.deepcopy(inDF)
    cliffInd = []

    # Convert pandas dataframe to numpy array
    data = inDF.iloc[:,1:].values

    # Set up spectral clustering model
    cluster_model = skc.SpectralClustering(n_clusters=10,
                                           random_state=42,
                                           n_jobs=7)

    # Apply spectral clustering
    classes = cluster_model.fit_predict(data)
    inDataCpy.iloc[:,0] = classes

    # Use classification MODI
    MODIVal,cliffInd = cMODI(inDataCpy)

    return MODIVal,cliffInd

def rMODI_DB(inDF,plot=False):
    '''
    Regression MODI using DBScan to classify each activity value.

    INPUT
        inDF: (pandas Data Frame) Input data frame with only the activity values.
        plot: (boolean) Decide whether to plot resulting groups.

    OUTPUT
        MODIVal: (float) MODI value.
    '''

    # Variables
    inDataCpy = copy.deepcopy(inDF)

    # Convert pandas dataframe to numpy array
    data = inDF.values

    # Calculate nearest neighbor indices
    #ind = calcNN(data)

    # Get column of regression values to classify
    regCol = np.reshape(data[:,0],(len(data[:,0]),1))

    # Choose value of epsilon based on average of kth-NN distance
    dist,ind = calcNN(regCol,knn=15)
    eps = np.mean(np.transpose(dist)[-1])

    # Set up DBScan
    db = skdb.DBSCAN(eps=eps,min_samples=5).fit(regCol)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Set up dataframe with new classifiers
    for index in range(len(regCol)):
        inDataCpy.iat[index,0] = labels[index]

    # Remove outlier rows
    inDataCpy = inDataCpy[(inDataCpy['Ratio Potency (uM)'] >= 0)]

    # Calculate cMODI
    rMODI = cMODI(inDataCpy)

    # Create set for each cluster
    unique_labels = set(labels)

    # Plot for viewing classification
    if (plot):
        # Define colors
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        # Plot data by group
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = regCol[class_member_mask & core_samples_mask]
            plt.plot(xy[:], np.ones(len(xy)), 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8)

            xy = regCol[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:], np.ones(len(xy)), 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        plt.xlabel(r'Concentration $(\mu M)$')
        plt.title('Cluster by Potency')
        plt.show()

    return rMODI

def cMODI(inDF):
    '''
    Classification MODI.

    INPUT
        inDF: (pandas Data Frame) Input data frame with activity values in first column and the rest populated by descriptors.

    OUTPUT
        MODIVal: (float) MODI value.

        cliffInd: (list of ints) List of indices of compounds which contribute to cliffs.
    '''

    # Variables
    activityDict = {}
    MODI = 0
    cliffInd = []

    # Convert pandas dataframe to numpy array
    data = inDF.values

    # Calculate nearest neighbor indices
    dist,ind = calcNN(data)

    # Determine activity cliffs
    for index,compound in enumerate(data):
        # Convert activity to integer
        activity = int(compound[0])

        # Create new activity if necessary
        if (activity not in activityDict):
            activityDict[activity] = [0,0]

        # Get nearest neighbor activity
        nnActivity = int(data[ind[index][0]][0])

        # Check for cliff
        if (nnActivity == activity):
            activityDict[activity][1] += 1
        else:
            cliffInd.append(index)

        # Update total
        activityDict[activity][0] += 1

    # Calculate MODI
    totalAct = len(activityDict)
    for key in activityDict:
        MODI += 1.0*activityDict[key][1]/activityDict[key][0]

    MODI *= (1.0/totalAct)

    return MODI,cliffInd

# Main
if (__name__ == '__main__'):
    '''
    Calculate MODI for given data.
    '''

    # Load and prep data file
    cData,rData = prepFile(inFileName)

    # Calculate classification MODI
    cMODI_Val = cMODI(cData)

    # Calculate regression MODI
    rMODI_Val = rMODI(rData)

    print('cMODI: {0}'.format(cMODI_Val))
    print('rMODI: {0}'.format(rMODI_Val))
