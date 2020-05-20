'''
Class for knn Read Across
'''

# Imports
import sklearn.neighbors as skneighbors
import numpy as np
import pandas as pd

class knnRARegressor:
    '''
    KNN Read Across Regressor.
    '''

    def __init__(knn=3):
        '''
        Set initial values.

        INPUT
            knn: (int) Number of nearest neighbors to consider.

        OUTPUT

        '''

        # Set values
        self.knn = knn

    def similarity(TrainDF,TestDF,TestIdx,method='tanimoto'):
        '''
        Calculate similarity using descriptors according to a specific method.

        INPUT
            TrainDF: (pandas Data Frame) Training data.

            TestDF: (pandas Data Frame) Testing data.

            TestIdx: (list of numpy arrays) Index list for nearest neighbors.

            method: (str) Method to use for calculating similarity.

        OUTPUT
            outSim: (list) Similarity for all compounds in TestIdx.

        NOTES
            Input dataframes should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.

        REFERENCES
            Willett, Peter, John M. Barnard, and Geoffrey M. Downs. "Chemical similarity searching." Journal of chemical information and computer sciences 38.6 (1998): 983-996.
        '''

        # Variables
        TrainDF_copy = TrainDF.copy()
        TestDF_copy = TestDF.copy()
        outSim = []

        # Only implemented for tanimoto coefficient
        if (method != 'tanimoto'):
            raise RuntimeError("Similarity calculation only implemented for Tanimoto coefficient.")

        # Loop over compounds
        for index,nnList in enumerate(TestIdx):
            dmySimList = []
            # Loop over nearest neighbors
            # Note that nnList contains the test index as the first element
            for nn in nnList[0][1:]:
                # Get descriptors
                xA = TestDF_copy.iloc[index,1:].values
                xB = TrainDF_copy.iloc[nn,1:].values

                # Calculate numerator elements
                num1 = xA*xB

                # Calculate denominator elements
                denom1 = xA*xA
                denom2 = xB*xB

                # Calculate similarity
                simVal = 1.0*np.sum(num1)/np.sum(denom1+denom2-num1)
                dmySimList.append(simVal)

            # Add similarity values to outSim
            outSim.append(dmySimList)

        return outSim

    def fit(TrainDF):
        '''
        Fit the KNN RA model using the given set of training data. There is technically no fitting process for KNN RA so the training data is just stored.

        INPUT
            TrainDF: (pandas Data Frame) Data frame containing training data.

        OUTPUT
        '''

        # Convert to pandas data frame if input is numpy array
        if (type(TrainDF) == np.ndarray):
            TrainDF_copy = pd.DataFrame(TrainDF)
            self.TrainingDF = TrainDF_copy.copy()
        else:
            self.TrainingDF = TrainDF.copy()

    def predict(TestDF):
        '''
        Determine activities using k-nearest neighbors read across.

        INPUT
            TestDF: (pandas Data Frame) Testing data.

        OUTPUT
            Y_Test_Pred: (numpy array) Numpy array containing predicted values.

        NOTES
            Input dataframes should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.

        REFERENCES
            Willett, Peter, John M. Barnard, and Geoffrey M. Downs. "Chemical similarity searching." Journal of chemical information and computer sciences 38.6 (1998): 983-996.
        '''

        # Convert to pandas data frame if input is numpy array
        if (type(TrainDF) == np.ndarray):
            TestDF_copy = pd.DataFrame(TestDF)
        else:
            TestDF_copy = TestDF.copy()

        # Variables
        numTestCompounds = (TestDF.shape)[0]
        indexList = []
        Y_Test_Pred = np.zeros(numTestCompounds)

        # Make sure testing and training sets have same number of descriptors
        if ((TestDF.shape)[1] != (self.TrainDF.shape)[1]):
            raise RuntimeError("Testing and Training compounds must have the same number of descriptors for knn read-across.")

        # Determine nearest neighbor indices for each test compound
        for testCmpd in range(numTestCompounds):
            # Add test compound to training data
            TrainDF_copy = self.TrainDF.copy()
            TrainDF_copy = TrainDF_copy.append(TestDF_copy.iloc[testCmpd,:])

            # Calculate indices
            tree = skneighbors.KDTree(TrainDF_copy.iloc[:,1:], leaf_size=2)
            dist, ind = tree.query([TrainDF_copy.iloc[-1,1:]], k=self.knn+1)

            # Store indices
            indexList.append(ind)

        # Calculate similarity
        TrainDF_copy = self.TrainDF.copy()
        simList = similarity(TrainDF_copy,TestDF_copy,indexList)

        # Determine activity
        for testCmpd in range(numTestCompounds):
            numerator = 0
            denominator = 0
            # Loop over nearest neighbors
            for index,simVal in enumerate(simList[testCmpd]):
                idxB = indexList[testCmpd][0][index+1]
                actB = TrainDF_copy.iloc[idxB,0]
                numerator += simVal*actB
                denominator += simVal

            Y_Test_Pred[testCmpd] = 1.0*numerator/denominator

        # Check accuracy
        # Y_Test_Pred = np.asarray(Y_Test_Pred)
        # Y_Test = TestDF.iloc[:,0].values
        # difference = Y_Test-Y_Test_Pred
        #
        # print(skmet.r2_score(Y_Test,Y_Test_Pred))

        return Y_Test_Pred
