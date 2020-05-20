'''
Model class

DESCRIPTION
    Class which holds all information and methods relevant to a model.
'''

# Imports
import copy
import pickle
import pandas as pd
import numpy as np
import os, sys
from ..Model import models
from ..Input import readInput as readIn
from ..Data import curate as curate
from ..Data import descriptors as descr
from ..Data import postprocess as pproc
from ..Validation import modelability as modi
from ..Validation import appDom as appDom

class Model:
    '''
    Class which contains information and methods relating to a QSAR model.
    '''

    # Constructor
    def __init__(self,inputParams={}):
        # Variables
        self.paramList = []     # List to store model parameters for each run

        # Set up model parameters
        self.modelParams = {'inputParams':inputParams}

        # Create new model_variables class
        #self.paramList.append(model_variables(inputParams))

    # Shallow copy
    def __copy__(self):
        return model(self.name)

    # Deep copy
    def __deepcopy__(self, memo):
        return model(copy.deepcopy(self.name, memo))

    # Load DescDF
    def load_Descriptors(self,modelParams):
        '''
        Load descriptor data frame from file.

        INPUT
            modelParams: (dict) Model parameters.

        OUTPUT
        '''

        # Load data
        fileName = self.modelParams['inputParams']['read_csv_desc']
        self.modelParams['DescDF'] = readIn.read_CSV(fileName)

    # Load inDF
    def load_Input(self,modelParams,quiet=False):
        '''
        Load input (activity,structure) data frame from file for training. If testing, only the structures are loaded.

        INPUT
            modelParams: (dict) Model parameters.

        OUTPUT
        '''

        # Print start message
        if (not quiet):
            print("========================================")
            print("Read Input File")

        # Load data
        fileName = self.modelParams['inputParams']['csvfilename']
        self.modelParams['inDF'] = readIn.read_CSV(fileName)
        self.modelParams['workingDF'] = self.modelParams['inDF'].copy()

        # Print number of compounds
        if (not quiet):
            print("\tNumber of Compounds: " + str((self.modelParams['inDF'].shape)[0]))

        # Print end
        if (not quiet):
            print("========================================")
            print("")

    # Data curation
    def data_curation(self,quiet=False):
        '''
        Perform data curation operations on workingDF.

        INPUT

        OUTPUT
        '''

        # Print start message
        if (not quiet):
            print("========================================")
            print("Data Curation")

        # Variables
        strcolname = self.modelParams['inputParams']["strcolname"]
        filter_atnum = self.modelParams['inputParams']["filter_atnum"]

        # Remove empty rows
        if (self.modelParams['inputParams']["rm_empty_rows"].lower() == 'true'):
            self.modelParams['workingDF'] = curate.removeEmptyRows(self.modelParams['workingDF'])

            if (not quiet):
                print("\tNumber of Empty Rows Removed: " + str((self.modelParams['inDF'].shape)[0]-(self.modelParams['workingDF'].shape)[0]))

        # Remove duplicates
        if (self.modelParams['inputParams']["rm_duplicates"].lower() == 'true'):
            startNum = (self.modelParams['workingDF'].shape)[0]
            self.modelParams['workingDF'] = curate.removeDuplicateStructures(self.modelParams['workingDF'],colName=strcolname)
            endNum = (self.modelParams['workingDF'].shape)[0]

            if (not quiet):
                print("\tNumber of Duplicates Removed: " + str(startNum-endNum))

        # Remove invalid SMILES
        if (self.modelParams['inputParams']["rm_invalid"].lower() == 'true'):
            startNum = (self.modelParams['workingDF'].shape)[0]
            self.modelParams['workingDF'] = curate.removeInvalidSmiles(self.modelParams['workingDF'],colName=strcolname)
            endNum = (self.modelParams['workingDF'].shape)[0]

            if (not quiet):
                print("\tNumber of Invalid SMILES Removed: " + str(startNum-endNum))

        # Remove salts
        if (self.modelParams['inputParams']["rm_salts"].lower() == 'true'):
            startNum = (self.modelParams['workingDF'].shape)[0]
            self.modelParams['workingDF'] = curate.removeSalts(self.modelParams['workingDF'],colName=strcolname)
            endNum = (self.modelParams['workingDF'].shape)[0]

            if (not quiet):
                print("\tNumber of Salts Removed: " + str(startNum-endNum))

        # Filter elements
        startNum = (self.modelParams['workingDF'].shape)[0]
        self.modelParams['workingDF'] = curate.filterElements(self.modelParams['workingDF'],
                                                              keepEle=filter_atnum,
                                                              colName=strcolname)
        endNum = (self.modelParams['workingDF'].shape)[0]

        if (not quiet):
            print("\tNumber of Compounds Removed by Element Filtering: " + str(startNum-endNum))

        # Print end
        if (not quiet):
            print("========================================")
            print("")

    # Calculate structures
    def calculate_structures(self):
        '''
        Calculate structures from structural information in workingDF.

        INPUT

        OUTPUT
        '''

        # Variables
        strcolname = self.modelParams['inputParams']["strcolname"]

        # Generate 3d structures
        if (self.modelParams['inputParams']["calc_3d"].lower() == 'true'):
            print('Calculating 3d Coordinates...')
            self.modelParams['workingDF'] = curate.smi2sdf_par(self.modelParams['workingDF'],colName=strcolname)

    # Calculate descriptors
    def calculate_descriptors(self):
        '''
        Calculate descriptors from structural information in inDF.

        INPUT

        OUTPUT
        '''

        # Variables
        strcolname = self.modelParams['inputParams']["strcolname"]

        # Set dimensionality
        if (self.modelParams['inputParams']["calc_3d"].lower() == 'true'):
            print('Calculating 3d Descriptors...')
            coord = 3
            colName = 'SDF'
        else:
            print('Calculating 2d Descriptors...')
            coord = 2
            colName = strcolname

        # Calculate descriptors
        self.modelParams['DescDF'] = descr.calc_mordred(self.modelParams['workingDF'],colName=colName,coord=coord)

        # Clean descriptors for training
        if (len(self.paramList) == 0):
            print('Cleaning Descriptors...')
            self.modelParams['DescDF'] = descr.cleanDescriptors(self.modelParams['DescDF'],descStart=coord)

        # Remove structure columns to prepare for modeling
        print('Removing structure columns...')
        if (coord == 3):
            self.modelParams['DescDF'] = self.modelParams['DescDF'].drop(labels=[strcolname,'SDF'],axis=1)
        else:
            self.modelParams['DescDF'] = self.modelParams['DescDF'].drop(labels=[strcolname],axis=1)

    # Curate descriptors
    def descriptor_curation(self):
        '''
        Curate features.

        INPUT

        OUTPUT
        '''

        # Imports
        import sklearn.preprocessing as skp

        # Only curate if training
        if (len(self.paramList) == 0):
            # Remove features with low standard deviation
            print('Removing features with low standard deviation...')
            for std in self.modelParams['inputParams']["low_std"]:
                self.modelParams['DescDF'] = descr.removeSTD(self.modelParams['DescDF'],thresh=std)

            # Remove correlated descriptors
            print('Removing correlated descriptors...')
            for corr in self.modelParams['inputParams']["corr_desc"]:
                self.modelParams['DescDF'] = descr.removeCorrelated(self.modelParams['DescDF'],thresh=corr)

            # Normalize descriptors
            print('Normalizing descriptors...')
            normDesc,self.modelParams['norms'] = skp.normalize(self.modelParams['DescDF'].iloc[:,1:],axis=0,return_norm=True)

            for index,colName in enumerate(self.modelParams['DescDF'].columns):
                # Skip activity column
                if (index == 0):
                    continue

                self.modelParams['DescDF'][colName] = normDesc[:,index-1]
                print('training!')
                print(self.modelParams['DescDF'][colName])

        # Match descriptors for testing
        if (len(self.paramList) >= 1):
            print('Matching Descriptors for Training...')

            # Determine columns to remove
            train_colNames = (self.paramList[0]['DescDF'].columns.values)[1:]
            test_colNames = (self.modelParams['DescDF'].columns.values)[1:]
            rmCols = [colName for colName in test_colNames if colName not in train_colNames]

            # Remove columns
            self.modelParams['DescDF'] = self.modelParams['DescDF'].drop(labels=rmCols,axis=1)

            # Normalize descriptors according to training norms
            print('Normalizing descriptors...')

            for index,colName in enumerate(self.modelParams['DescDF'].columns):
                # Skip activity column
                if (index == 0):
                    continue

                self.modelParams['DescDF'][colName] = self.modelParams['DescDF'][colName]/self.paramList[0]['norms'][index-1]
                print('testing!')
                print(self.modelParams['DescDF'][colName])
    # Calculate training set applicability domain


    # Modelability
    def calculate_MODI(self):
        '''
        Calcualate modelability and possibly remove activity cliffs.

        INPUT

        OUTPUT
        '''

        # Calculate MODI
        if (self.modelParams['inputParams']["model_type"] == "classification"):
            self.modelParams['MODIVal'], self.modelParams['cliffIdx'] = modi.cMODI(self.modelParams['DescDF'])
            print('Classification MODI: ' + str(self.modelParams['MODIVal']))
        else:
            self.modelParams['MODIVal'], self.modelParams['cliffIdx'] = modi.rMODI_Spectral(self.modelParams['DescDF'])
            print('Regression MODI (Spectral): ' + str(self.modelParams['MODIVal']))

        # Remove cliffs
        if (self.modelParams['inputParams']["rm_modi"].lower() == "true"):
            # Save information about full descriptors
            self.modelParams['DescDF_FullDesc'] = self.modelParams['DescDF'].copy()
            self.modelParams['MODIVal_FullDesc'] = self.modelParams['MODIVal']
            self.modelParams['cliffIdx_FullDesc'] = copy.deepcopy(self.modelParams['cliffIdx'])

            # Remove cliffs
            print('Removing ' + str(len(self.modelParams['cliffIdx'])) + ' compounds for MODI...')
            self.modelParams['DescDF'] = self.modelParams['DescDF'].drop(self.modelParams['DescDF'].index[self.modelParams['cliffIdx']])

            # Compute new MODI
            self.modelParams['MODIVal'], self.modelParams['cliffIdx'] = modi.cMODI(self.modelParams['DescDF'])
            print('New MODI: ' + str(self.modelParams['MODIVal']))

    # Fit model
    def fit_model(self):
        '''
        Fit model.

        INPUT

        OUTPUT
        '''

        # Imports
        import sklearn.model_selection as skm
        import sklearn.decomposition as skd

        # Try PCA
        '''
        ncomp = 20
        pca = skd.PCA(n_components=ncomp)
        pca.fit(np.transpose(self.modelParams['DescDF'].iloc[:,1:].values))
        self.modelParams['DescDF'].iloc[:,1:ncomp+1] = np.transpose(pca.components_)
        self.modelParams['DescDF'] = self.modelParams['DescDF'].drop(labels=self.modelParams['DescDF'].columns.values[ncomp+1:],axis=1)
        '''

        # Perform clustering
        #modi.show_hierarchical_clustering(self.modelParams['DescDF'])

        # Split data
        self.modelParams['trainDF'],self.modelParams['testDF'] = skm.train_test_split(self.modelParams['DescDF'],
                                                                                      test_size=self.modelParams['inputParams']["test_split"],
                                                                                      random_state=42)

        print("Total number of compounds: " + str((self.modelParams['DescDF'].shape)[0]))

        # Fit model
        if (self.modelParams['inputParams']["model_type"] == "regression"):
            print('Regression...')

            # Scikit learn random forest | regression
            if (self.modelParams['inputParams']["model"] == "random_forest"):
                print("--Random Forest--")
                self.modelParams['Fit_Pred_Train'],self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Test'],self.modelParams['Fit_Test'],self.modelParams['model_Fit'] = models.model_rf_reg(self.modelParams['trainDF'],self.modelParams['testDF'])

            # Scikit learn neural network | regression
            elif (self.modelParams['inputParams']["model"] == "neural_network"):
                print("--Neural Network--")
                self.modelParams['Fit_Pred_Train'],self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Test'],self.modelParams['Fit_Test'],self.modelParams['model_Fit'] = models.model_nn_reg(self.modelParams['trainDF'],self.modelParams['testDF'])

            # KNN Read across | regression
            elif (self.modelParams['inputParams']["model"] == "knn_ra"):
                print("--KNN Read Across--")

                # Set training prediction to empty list
                self.modelParams['Fit_Pred_Train'] = []

                self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Test'],self.modelParams['Fit_Test'],self.modelParams['model_Fit'] = models.model_knnra_reg(self.modelParams['trainDF'],self.modelParams['testDF'],knn=2)
        else:
            print('Classification...')

            # Scikit learn random forest | classification
            if (self.modelParams['inputParams']["model"] == "random_forest"):
                print("--Random Forest--")
                self.modelParams['Fit_Pred_Train'],self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Test'],self.modelParams['Fit_Test'],self.modelParams['model_Fit'] = models.model_rf_class(self.modelParams['trainDF'],self.modelParams['testDF'])

        # Post processing
        if (self.modelParams['inputParams']["postproc"].lower() == "true"):
            fitParams = pproc.pca_shift_init(self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Train'],plot=False)
            self.modelParams['Fit_Pred_Train_Shift'],r_value_train = pproc.pca_shift_calc(self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Train'],fitParams,plot=True)

            # Only apply if test set is not empty
            if (len(self.modelParams['Fit_Test']) != 0):
                self.modelParams['Fit_Pred_Test_Shift'],r_value_test = pproc.pca_shift_calc(self.modelParams['Fit_Test'],self.modelParams['Fit_Pred_Test'],fitParams,plot=True)
            else:
                self.modelParams['Fit_Pred_Test_Shift'] = []

            # Plot final regression
            models.plotPrediction(self.modelParams['Fit_Train'],self.modelParams['Fit_Pred_Train_Shift'],self.modelParams['Fit_Test'],self.modelParams['Fit_Pred_Test_Shift'])
        else:
            self.modelParams['Fit_Pred_Train_Shift'] = []
            self.modelParams['Fit_Pred_Test_Shift'] = []

    # Save model
    def save_model(self,outFileName='model.pickle'):
        '''
        Save model.

        INPUT
            outFileName: (str) Name of output file.

        OUTPUT
        '''

        # Save model class
        with open(outFileName,'wb') as outFile:
            pickle.dump(self,outFile)

    # Train model
    def train_model(self):
        '''
        Train model by performing all data and descriptor curation.

        INPUT

        OUTPUT
        '''

        # Variables
        loaded_descr = False

        # Read input file
        if (len(self.modelParams['inputParams']["read_csv_desc"]) > 0):
            print('Loading descriptors...')
            self.load_Descriptors(self.modelParams)
            loaded_descr = True
        else:
            print("Reading input file...")
            self.load_Input(self.modelParams)

        # Work on data if not provided with descriptor file
        if (loaded_descr == False):
            # Data curation
            self.data_curation()

            # Calculate structures
            self.calculate_structures()

            # Calculate descriptors
            self.calculate_descriptors()

            # Curate descriptors
            self.descriptor_curation()

            # Save descriptors
            if (self.modelParams['inputParams']["save_csv"].strip() != ""):
                self.modelParams['DescDF'].to_csv(self.modelParams['inputParams']["save_csv"],index=False)

        # Descriptor curation when loading descriptor files
        if ((loaded_descr == True) and (self.modelParams['inputParams']["curate_desc"].lower() == "true")):
            self.descriptor_curation()

        # Calculate MODI
        self.calculate_MODI()

        # Fit model
        self.fit_model()

        # Write csv of results
        #outDF = pd.DataFrame(self.modelParams['Fit_Pred_Train'])
        #result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])

        # Store results
        self.paramList.append(self.modelParams)
        save_model = self.modelParams['inputParams']["save_model"]
        self.modelParams = {}

        # Save model
        if (save_model.strip() != ""):
            self.save_model(outFileName=save_model)

    # Test model
    def test_model(self,inputParameters=None):
        '''
        Test model on set of data.

        INPUT

        OUTPUT
        '''

        # Initialize model parameters
        self.modelParams = {'inputParams':inputParameters}

        # Load file
        self.load_Input(self.modelParams)

        # Data curation
        self.data_curation()

        # Calculate structures
        self.calculate_structures()

        # Calculate descriptors
        self.calculate_descriptors()

        # Curate descriptors
        self.descriptor_curation()

        # Save descriptors
        if (self.modelParams['inputParams']["save_csv"].strip() != ""):
            self.modelParams['DescDF'].to_csv(self.modelParams['inputParams']["save_csv"],index=False)

        # Test model
        modelFit = self.paramList[0]['model_Fit']
        Y_Pred,X_Test = models.model_test(self.modelParams['DescDF'],modelFit)

        # Save results
        saveDF = pd.DataFrame()
        saveDF['predict'] = Y_Pred
        saveDF['SMILES'] = self.modelParams['workingDF']['SMILES'].values
        saveDF.to_csv('prediction.csv',index=False)

if (__name__ == '__main__'):
    pass
