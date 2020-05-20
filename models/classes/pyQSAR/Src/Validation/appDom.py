'''
Methods for analyzing the applicability domain of QSAR models.

DESCRIPTION
    This module holds functions for examining the applicability domain for QSAR models.

!!! IN DEVELOPMENT !!!
'''

# Imports
import numpy as np
import scipy.stats as scst
import multiprocessing as mp

def ad_pdf_normal_kernel_1(index1,X):
    '''
    Kernel 1 for MND.
    '''

    # Variables
    numCompounds = (X.shape)[0]
    dataArray = np.zeros(numCompounds)
    #print('Index: ' + str(index1))

    # Initial query data position
    pos1 = X[index1,:]

    for index2 in range(index1,numCompounds):
        # Skip onsite interactions
        if (index1 == index2):
            continue

        # Get source data position
        pos2 = X[index2,:]

        # Smoothing / Covariance
        #Sigma = 100*2271.89
        Sigma = 0.1

        # Calculate pdf
        F = scst.multivariate_normal(pos2, Sigma)
        dataArray[index1] += F.pdf(pos1)
        dataArray[index2] += F.pdf(pos1)

    return dataArray

def ad_pdf_normal(testingDF,trainingDF):
    '''
    Uses multivariate normal distribution to analyze applicability domain.

    INPUT
        testingDF: (pandas Data Frame) Data frame of testing points.

        trainingDF: (pandas Data Frame) Data frame of training points.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with testing compounds removed which were outside of the applicability domain.
    '''

    # Variables
    outDF = testingDF.copy()

    # Separate out data
    X = (trainingDF.values)[:,1:]
    Y = (trainingDF.values)[:,0]
    X_Test = (testingDF.values)[:,1:]
    Y_Test = (testingDF.values)[:,0]

    # Calculate pdf at each training point
    print('\tCalculating pdf for training points...')
    dataArray = np.zeros(len(Y))

    # Pack for mp
    parPack = []

    for index1 in range((X.shape)[0]):
        parPack.append((index1,X))

    # Parallel processing
    print('\tRunning Parallel!')
    with mp.Pool(processes=2) as pool:
        result = pool.starmap(ad_pdf_normal_kernel_1,parPack)

    # Add parallel results together
    for dataResult in result:
        dataArray += dataResult

    cutoff_95 = np.percentile(dataArray,5)
    cutoff_97 = np.percentile(dataArray,3)
    cutoff_99 = np.percentile(dataArray,1)
    cutoff_99_99 = np.percentile(dataArray,100-99.9999999)

    print('95: ' + str(cutoff_95))
    print('99: ' + str(cutoff_99))

    # Determine where testing compounds are in the pdf
    print('\t Determining AD for testing points...')
    dataArray_Test = np.zeros(len(Y_Test))

    ## Loop over training compounds
    for index1 in range(len(Y)):
        #print('Testing: ' + str(index1))
        # Initial query data position
        pos1 = X[index1,:]

        # Smoothing / Covariance
        #Sigma = 100*2271.89
        Sigma = 0.1

        # Calculate pdf
        F = scst.multivariate_normal(pos1, Sigma)

        ## Loop over testing compounds
        for index2 in range(len(Y_Test)):
            # Get source data position
            pos2 = X_Test[index2,:]

            # PDF addition
            dataArray_Test[index2] += F.pdf(pos2)

    # Determine compounds to remove outside of necessary percentile
    rmIdx = []
    #print('\tLength of data array test ' + str(len(dataArray_Test)))

    for index,pdfVal in enumerate(dataArray_Test):
        if (pdfVal <= cutoff_95):
            rmIdx.append(index)

    # Remove compounds
    print('\tRemoving ' + str(len(rmIdx)) + ' compounds...')
    outDF = outDF.drop(outDF.index[rmIdx])

    return outDF

if (__name__ == '__main__'):
    pass
