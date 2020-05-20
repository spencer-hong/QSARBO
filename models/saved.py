import GPy, GPyOpt
import numpy as np
import pandas as pd
import sklearn.model_selection as mose
import sklearn.preprocessing as skp
import sklearn.ensemble as sken
import sklearn.neighbors as skne
import sklearn.metrics as me 
from tensorflow.keras.models import load_model

import json
import pickle

from datetime import datetime
import sys, os

sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classes import prepare as prepare

del sys.path[0]

import random
import gc
gc.collect()


def saved(datastore, models):

	dataname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]
	# if filename:
	#     with open(filename, 'r') as f:
	#         datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = 'm', filelocation = dataname, chemID = datastore["chemID"]["content"])
	selected_data = selected_data.drop('m', axis = 1)
	print("-----------------------------------")
	print("Cleaning Data")
	print("-----------------------------------\n")
	inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
	print("-----------------------------------")
	print("Curating Descriptors")
	print("-----------------------------------\n")
	print(f"Number of Compounds: {inDF.shape[0]}")
	inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold = 2, STDthreshold = 0, IDboolean = IDboolean)
	#activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF = prepare.partition(df = inDF,validset =  datastore['valid_split']['content'], testset = datastore['test_split']['content'], IDboolean = IDboolean)
	#print(inDF.head)
	Name = inDF.loc[:,datastore["column_SMILES"]['content']]
	inDF = inDF.drop(datastore["column_SMILES"]['content'], axis = 1)
	#dfdict  = pickle.load( open( "rf/descriptors.p", "rb" ) )
	#trainDF = dfdict['trainDF']
	#X_Train = trainDF
	#print(inDF.head)
	dfdict  = pickle.load( open( current_folder + "pickled/rf_descriptors.p", "rb" ) )
	rf_cols = dfdict["rf_cols"]
	newrf_cols = rf_cols[2:]
	newinDF = inDF.loc[:, newrf_cols]
	#print(newinDF.head)
	#newinDF = newinDF.drop(datastore['column_SMILES']['content'], axis = 1)
	#newinDF = newinDF.drop(datastore['column_activity']['content'], axis = 1)
	print("-----------------------")
	print("Loading Random Forest Model")
	print("-----------------------")
	print(newinDF)
	rf_model  = pickle.load( open(current_folder +  "pickled/rfmodel.p", "rb" ) )
	y_pred_rf = rf_model.predict(newinDF) #took out newinDF here
	del(rf_model)
	del(selected_data)

	shapenum = inDF.shape[0]

	print("-----------------------")
	print("Loading Neural Network Model")
	print("-----------------------")
	#rf_model = rfdict['model']
	#nndict = pickle.load(open("saved/nnmodel.p", "rb"))
	nn_model = load_model('pickled/nnmodel.h5')

	dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
	nn_cols = dfdict["nn_cols"]
	newnn_cols = nn_cols[2:]
	newinDF = inDF.loc[:, newnn_cols]
	y_pred_nn = nn_model.predict(newinDF)
	print(y_pred_nn)
	del(nn_model)

	if datastore['chemID']['content'] == 'NA':
		IDboolean = False
	else:
		IDboolean = True

	if IDboolean:
		SMILESList = []
		rflist = []
		NAMESList = []
		nn_list = []
		combi_list = []
		
		for i in range(0,shapenum):
			NAMESList.append(Name.loc[:,].values[i])
			SMILESlist.append(SMILES.loc[:,].values[i])
			rf_list.append(y_pred_rf[i])
			nn_list.append(y_pred_nn[i][0])

		combidict = pickle.load(open("saved/combi.p", "rb"))

		y_pred_combi = np.array(nn_list) * combidict["best weight nn"]  + np.array(rf_list)*combidict["best weight rf"]
		res = pd.DataFrame({'SMILES':SMILESList, 'Chemical ID': NAMESList, 'Random Forest Prediction':rf_list, 'Neural Network Prediction':nn_list, 'Combi-QSAR Prediction':y_pred_combi})
	else:
		SMILESList = []
		rf_list = []
		nn_list = []
		combilist = []
		
		for i in range(0,shapenum):
			SMILESList.append(Name.loc[:,].values[i])
			rf_list.append(y_pred_rf[i])
			nn_list.append(y_pred_nn[i][0])

		combidict = pickle.load(open("pickled/combi.p", "rb"))
		#print(rf_list)
		#print('--------------------')
		y_pred_combi = np.array(nn_list) * float(combidict["best weight nn"])  + np.array(rf_list)*float(combidict["best weight rf"])
		#print(nn_list)
		#print(y_pred_combi)
		res = pd.DataFrame({'SMILES':SMILESList,  'Random Forest Prediction':rf_list, 'Neural Network Prediction':nn_list, 'Combi-QSAR Prediction':y_pred_combi})

	del(inDF)
	res.to_csv('predictions/unknowntested.csv', sep=',')
	print("-----------------------")
	print("Saving Predictions")
	print("-----------------------")
	return "done"
		