import numpy as np 
import pandas as pd
import sklearn.model_selection as mose
import sklearn.preprocessing as skp
# import pyQSAR descriptor curation methods


import sys, os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/classes')

from pyQSAR.Src.Data import descriptors as descr
from pyQSAR.Src.Data import curate
del sys.path[0]

def isolate( filelocation, structname = 'SMILES', activityname = 'ACTIVITY',chemID='NA' ):
	"""
	takes in CSV file containing structure column (specified) and activity column
	
	chemicalidentifier is the name of the column that contains chemical name/CAS/non-SMILES identifer for the dataset that you would like to keep to identify the chemical after this analysis.

	returns a pandas dataframe containing only the structure column and activity column
	"""
	rawdata = pd.read_csv(filelocation)

	if chemID == 'NA':
		IDboolean = False
		#only select the activity and SMILES
		selected_data = rawdata.loc[:, [activityname, structname ]]
	else:
		IDboolean = True
		selected_data = rawdata.loc[:, [activityname, chemID, structname ]]


	return selected_data, IDboolean

def cleanSMILES(df, elementskept, smilesName):
	"""
	takes in pandas dataframe containing the structure and activity columns and deletes duplicate, invalid, or salts. It also filters elements based on a given list of atomic numbers. elementskept must be in a list. colName is the name of the SMILES column.

	returns a cleaned dataframe
	"""

	inDF = curate.removeDuplicateStructures(df, colName = smilesName)
	inDF = curate.removeInvalidSmiles(inDF, colName = smilesName)
	inDF = curate.removeSalts(inDF, colName = smilesName) 
	inDF = curate.filterElements(inDF, keepEle=elementskept, colName = smilesName)


	return inDF

def createdescriptors(df, colName, correlationthreshold = 0.95, STDthreshold = 0.25, IDboolean = False):
	"""
	takes in pandas dataframe and creates descriptors. Cleans descriptors based on correlation and standard deviation.

	returns a dataframe
	"""

	# transform into 3D structures
	print('Transforming SMILES to 3D Structures')
	inDF = curate.smi2sdf_par(df, colName = colName)
	# calculate descriptors using mordred
	print('Calculating Descriptors')
	inDF = descr.calc_mordred(inDF, colName = "SDF", coord=3)
	print('Removing Correlated Descriptors')
	# remove highly correlated structures
	inDF = descr.removeCorrelated(inDF, thresh = correlationthreshold)

	inDF = descr.removeSTD(inDF, thresh = STDthreshold)
	print('Cleaning Descriptors')
	if IDboolean:
		descStart = 3
	else:
		descStart = 2
	inDF = descr.cleanDescriptors(inDF, descStart = descStart)

	# removes 3D structure (no longer needed)
	inDF = inDF.drop('SDF', axis = 1)

	return inDF

def partition(IDboolean, df, validset = 0.1, testset = 0.2):
	"""
	takes in pandas dataframe

	validset is the percentage (in decimals) of the whole dataset you want to set aside for external validation

	testset is the percentage (in decimals) of the training dataset you want to set aside for internal validation

	returns a series of NORMALIZED dataframes: activity for training, activity for testing, activity for validation, chemical identifer (SMILES and nonSMILES if present) for training, testing, and validation, and descriptor dataframes for training, testing, and validation
	"""
	startdesc = 0
	trainDF, validDF = mose.train_test_split(df, test_size=validset)
	rf_cols = list(trainDF.columns.values)
	trainDF, testDF = mose.train_test_split(trainDF, test_size = testset)
	if IDboolean:
		startdesc = 3
		IDValidDF = validDF.iloc[:, 2]
		IDTrainDF = trainDF.iloc[:, 2]
		IDTestDF = testDF.iloc[:, 2]
		nameValidDF = validDF.iloc[:, 1]
		nameTrainDF = trainDF.iloc[:, 1]
		nameTestDF = testDF.iloc[:, 1]
		activityValidDF = validDF.iloc[:, 0]
		activityTrainDF = trainDF.iloc[:, 0]
		activityTestDF = testDF.iloc[:, 0]
		validDF = validDF.iloc[:, startdesc:]
		trainDF = trainDF.iloc[:, startdesc:]
		testDF = testDF.iloc[:, startdesc:]
	else:
		startdesc = 2
		IDValidDF = validDF.iloc[:, 1]
		IDTrainDF = trainDF.iloc[:, 1]
		IDTestDF = testDF.iloc[:, 1]
		nameValidDF = None
		nameTrainDF = None
		nameTestDF = None 
		activityValidDF = validDF.iloc[:, 0]
		activityTrainDF = trainDF.iloc[:, 0]
		activityTestDF = testDF.iloc[:, 0]
		validDF = validDF.iloc[:, startdesc:]
		trainDF = trainDF.iloc[:, startdesc:]
		testDF = testDF.iloc[:, startdesc:]
		#print(trainDF)
	scaler = skp.StandardScaler()
	trainDF = scaler.fit_transform( trainDF )
	testDF = scaler.transform( testDF )
	validDF = scaler.transform(validDF)

	return activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF, rf_cols

