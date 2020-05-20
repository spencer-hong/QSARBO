import GPy, GPyOpt
import numpy as np
import pandas as pd
import sklearn.model_selection as mose
import sklearn.preprocessing as skp
import sklearn.ensemble as sken
import sklearn.neighbors as skne
import sklearn.metrics as me 
from timeit import default_timer as timer
import matplotlib.pyplot as plt
#import ortools 

from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import json
import pickle
import warnings
import logging
#from datetime import datetime
import random

import sys, os

sys.path.insert(0,
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classes import prepare as prepare
from models.classes import nn as nn
from models.classes import nnc as nnc 
del sys.path[0]

random.seed(36)
import faulthandler; faulthandler.enable()


def runner_nn(input_file_loc, iterator = 0):

	logging.getLogger('tensorflow').disabled = True
	os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
	os.system('export KMP_WARNINGS=FALSE')
	warnings.filterwarnings("ignore", category=FutureWarning)
	dirName = 'pickled'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists. Skipping creation.")
	dirName = 'predictions'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists. Skipping creation.")

	# function to run random forest class
	def run_nn( X_Train , Y_Train, X_Test, Y_Test,  l1_out=512, 
					 l2_out=512, 
					 l1_drop=0.2, 
					 l2_drop=0.2, 
					 l3_out = 512,
					 l3_drop = 0.2,
					 batch_size=100, 
					 epochs=10, cv = 7):
		_nn = nn.nn(cv = cv, l1_out = l1_out, l2_out = l2_out, l3_out = l3_out, l1_drop = l1_drop, l2_drop = l2_drop, l3_drop = l3_drop, X_Train = X_Train, Y_Train = Y_Train, X_Test = X_Test, Y_Test = Y_Test)
		nn_evaluation = _nn.nn_evaluate()
		return nn_evaluation

	if input_file_loc:
		with open(input_file_loc, 'r') as f:
			datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]
	if datastore['saved_hyperparameters?']['content'] == "False":
		start = timer()
		
		if datastore['saved_descriptors?']['content'] == "False":
			selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename, chemID = datastore["chemID"]["content"])
			print("-----------------------------------")
			print("Cleaning Data")
			print("-----------------------------------\n")
			inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
			print("-----------------------------------")
			print("Curating Descriptors")
			print("-----------------------------------\n")
			print(f"Number of Compounds: {inDF.shape[0]}")
			inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold =  datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
			activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF, nn_cols = prepare.partition(df = inDF,validset =  datastore['valid_split']['content'], testset = datastore['test_split']['content'], IDboolean = IDboolean)
			print("-----------------------------------")
			print("Partitioning Data")
			print("-----------------------------------\n")
			dfdict = {
			"activityValidDF": activityValidDF,
			"activityTrainDF": activityTrainDF,
			"activityTestDF": activityTestDF,
			"IDValidDF":IDValidDF,
			"IDTrainDF":IDTrainDF,
			"IDTestDF":IDTestDF,
			"validDF":validDF,
			"trainDF":trainDF,
			"testDF":testDF,
			"nameValidDF":nameValidDF,
			"nameTrainDF":nameTrainDF,
			"nameTestDF":nameTestDF,
			"nn_cols":nn_cols
			}
			pickle.dump( dfdict, open( current_folder+ "pickled/nn_descriptors.p", "wb" ) )
		else:
			print("-----------------------------------")
			print("Loading Descriptors")
			print("-----------------------------------\n")
			dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
			activityValidDF = dfdict['activityValidDF']
			activityTrainDF = dfdict['activityTrainDF']
			activityTestDF = dfdict['activityTestDF']
			IDValidDF = dfdict['IDValidDF']
			IDTrainDF = dfdict['IDTrainDF']
			IDTestDF = dfdict['IDTestDF']
			validDF = dfdict['validDF']
			trainDF = dfdict['trainDF']
			testDF = dfdict['testDF']
			nameValidDF = dfdict['nameValidDF']
			nameTrainDF = dfdict['nameTrainDF']
			nameTestDF = dfdict['nameTestDF']

		X_Valid = validDF
		Y_Valid = activityValidDF
		X_Train = trainDF
		Y_Train = activityTrainDF
		X_Test = testDF
		Y_Test = activityTestDF
		orilen = X_Train.shape[1]
		bounds = [
				  {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
				  {'name': 'l2_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
				  {'name':'l3_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
				  {'name': 'l1_out',           'type': 'discrete',    'domain': (orilen/4, orilen/3, orilen/2, orilen, orilen*2, orilen*3,orilen*4)},
				  {'name': 'l2_out',           'type': 'discrete',    'domain': ( orilen/4,orilen/3, orilen/2, orilen, orilen*2, orilen*3,orilen*4)},
				   {'name': 'l3_out',           'type': 'discrete',    'domain': ( orilen/4,orilen/3, orilen/2, orilen, orilen*2,orilen*3, orilen*4)},
				  {'name': 'batch_size',       'type': 'discrete',    'domain': (200,300, 400,500, 600, 700)},
				  {'name': 'epochs',           'type': 'discrete',    'domain': (300, 350, 400, 450, 500, 600, 700)}]
		def f(x):
			evaluation = run_nn(
			l1_drop = float(x[:,0]), 
			l2_drop = float(x[:,1]), 
			l3_drop = float(x[:, 2]),
			l1_out = int(x[:,3]),
			l2_out = int(x[:,4]), 
			l3_out = int(x[:, 5]),
			batch_size = int(x[:,6]), 
			epochs = int(x[:,7]), X_Train = X_Train, Y_Train = Y_Train ,X_Test = X_Test, Y_Test =Y_Test, cv = datastore['cvfolds?']['content'])
			return evaluation
		print("-----------------------------------")
		print("Bayesian Optimization Initiated: First Picking 5 Random Sample Points")
		print("-----------------------------------\n")
		BOModel = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, maximize=True, num_cores = datastore['num_cores']['content'])
		print("-----------------------------------")
		print("Bayesian Optimization: Now Searching 20 Points")
		print("-----------------------------------\n")
		BOModel.run_optimization(max_iter=20)
		print("-----------------------------------")
		print("Bayesian Optimization Converged")
		print("-----------------------------------\n")
		print("-----------------------------------")
		print("Best Hyperparameters Found:\n")
		best_l1_drop = BOModel.x_opt[0]
		best_l2_drop = BOModel.x_opt[1]
		best_l3_drop = BOModel.x_opt[2]
		best_l1_out = BOModel.x_opt[3]
		best_l2_out = BOModel.x_opt[4]
		best_l3_out = BOModel.x_opt[5]
		best_batch_size = BOModel.x_opt[6]
		best_epochs = BOModel.x_opt[7]
		print(f"Layer 1 Drop: {best_l1_drop}")
		print(f"Layer 2 Drop: {best_l2_drop}")
		print(f"Layer 3 Drop: {best_l3_drop}")
		print(f"Layer 1 Neurons: {best_l1_out}")
		print(f"Layer 2 Neurons: {best_l2_out}")
		print(f"Layer 3 Neurons: {best_l3_out}")
		print(f"Batch Size: {best_batch_size}")
		print(f"Epochs: {best_epochs}")
		print("-----------------------------------\n")
		hypdict = {
			"l1_drop":best_l1_drop,
			"l2_drop":best_l2_drop,
			"l3_drop":best_l3_drop,
			"l1_out":best_l1_out,
			"l2_out":best_l2_out,
			"l3_out":best_l3_out,
			"batch_size":best_batch_size,
			"epochs":best_epochs}
		pickle.dump( hypdict, open(current_folder+ "pickled/nn_hyperparameters.p", "wb" ))
	else:
		start = timer()
		print("-----------------------------------")
		print("Loading Descriptors")
		print("-----------------------------------\n")
		dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
		activityValidDF = dfdict['activityValidDF']
		activityTrainDF = dfdict['activityTrainDF']
		activityTestDF = dfdict['activityTestDF']
		IDValidDF = dfdict['IDValidDF']
		IDTrainDF = dfdict['IDTrainDF']
		IDTestDF = dfdict['IDTestDF']
		validDF = dfdict['validDF']
		trainDF = dfdict['trainDF']
		testDF = dfdict['testDF']
		nameValidDF = dfdict['nameValidDF']
		nameTrainDF = dfdict['nameTrainDF']
		nameTestDF = dfdict['nameTestDF']

		X_Valid = validDF
		Y_Valid = activityValidDF
		X_Train = trainDF
		Y_Train = activityTrainDF
		X_Test = testDF
		Y_Test = activityTestDF

		print("-----------------------------------")
		print(f"Loading Optimized Parameters")
		print("-----------------------------------\n")
		hypdict  = pickle.load( open( current_folder + "pickled/nn_hyperparameters.p", "rb" ) )
		best_l1_drop = hypdict["l1_drop"]
		best_l2_drop = hypdict["l2_drop"]
		best_l3_drop = hypdict["l3_drop"]
		best_l1_out = hypdict['l1_out']
		best_l2_out = hypdict['l2_out']
		best_l3_out = hypdict['l3_out']
		best_batch_size = hypdict['batch_size']
		best_epochs = hypdict['epochs']
	print("-----------------------------------")
	print(f"Training Neural Network with Optimized Parameters")
	print("-----------------------------------\n")
	model = Sequential()
	model.add(Dense(int(best_l1_out), input_shape=(X_Train.shape[1], ), kernel_initializer = 'uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(best_l1_drop))
	model.add(Dense(int(best_l2_out), activation='relu',
						kernel_initializer = 'uniform'))
	model.add(Dropout(best_l2_drop))
	model.add(Activation('relu'))

	model.add(Dense(int(best_l3_out), activation = 'relu', kernel_initializer = 'uniform'))
	model.add(Dropout(best_l3_drop))
			#model.add(Dropout(self.l3_drop))
	#model.add(Dense(1,activation = 'relu'))
	model.add(Dense(1))
	model.add(Activation('linear'))
	model.compile(loss='mean_squared_error',
						  optimizer=Adam())

	#early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
			
	model.fit(X_Train, Y_Train,
						   batch_size=int(best_batch_size),
						   epochs=int(best_epochs),
						   verbose=1)

	print("-----------------------------------")
	print(f"Testing Neural Network with Optimized Parameters")
	print("-----------------------------------\n")
	y_pred = model.predict(X_Test)
	y_pred_train = model.predict(X_Train)
	y_pred_valid = model.predict(X_Valid)
	score_test = me.r2_score(Y_Test, y_pred)
	score_train = me.r2_score(Y_Train, y_pred_train)
	score_valid = me.r2_score(Y_Valid, y_pred_valid)

	model.fit(X_Test, Y_Test, batch_size = int(best_batch_size), epochs=int(best_epochs), verbose = 0)

	model.save(current_folder + 'pickled/nnmodel.h5')
	#pickle.dump(nnmodeldict, open("saved/nnmodel.p", "wb"))
	pickle.dump( score_valid, open( current_folder + "pickled/nn_validscore.p", "wb" ) )
	print("-----------------------------------")
	print(f"Final Results")
	print(f"Training R-squared: {score_train}")
	print(f"Testing R-squared: {score_test}")
	print(f"Validation R-squared: {score_valid}")
	print("-----------------------------------\n")
	end = timer()
	time_taken = end - start
	print(f"Time Taken: {time_taken} seconds")
	try:
		os.remove(current_folder + "tmpSDF.sdf")
	except FileNotFoundError:
		print("File Not Found")
	print("-----------------------------------")
	print("Saving Predictions...")
	print("-----------------------------------\n")

	if datastore['chemID']['content'] == 'NA':
		IDboolean = False
	else:
		IDboolean = True

	if IDboolean:
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		NAMESList = []
		for i in range(0,IDTestDF.shape[0]):
			NAMESList.append(nameTestDF.loc[:,].values[i])
			SMILESTest.append(IDTestDF.loc[:,].values[i])
			YTestList.append(Y_Test.loc[:,].values[i])
			YTestPredList.append(y_pred[i][0])
		for i in range(0,IDTrainDF.shape[0]):
			NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i][0])

		res = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i][0])

		res_valid = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
	else:
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		SMILESValid = []
		YValidList = []
		YValidPredList = []
		for i in range(0,IDTestDF.shape[0]):
			SMILESTest.append(IDTestDF.loc[:,].values[i])
			YTestList.append(Y_Test.loc[:,].values[i])
			YTestPredList.append(y_pred[i][0])
		for i in range(0,IDTrainDF.shape[0]):
			#NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i][0])

		res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		#NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			#NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i][0])
			
		res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})


	res.to_csv(current_folder + 'predictions/nn_test.csv', sep=',')
	res_valid.to_csv(current_folder + 'predictions/nn_valid.csv', sep=',')
	print("-----------------------------------")
	print("Neural Network Finished!")
	print("-----------------------------------\n")


	print("-----------------------------------")
	print("Time to do visualizations!")
	print("-----------------------------------\n")

	## df is a dataframe containing the smiles, actual, and prediction
	## returns the dataframe containing leverages
	def calculate_leverage(df):
		actualmean = df['Actual'].mean()
		num = df.shape[0]

		denom = 0
		for i in range(0, num):
			denom += (df['Actual'][i] - actualmean) ** 2.

		outside=[]
		leverage = []
		for i in range(0, num):
			leverage_i = ((df['Actual'][i] - actualmean)** 2.)/(denom)  + (1/num)
			leverage.append(leverage_i)
			if leverage_i > 0.012:
				outside.append('Invalid')
			else:
				outside.append('Valid')
		df.insert(2, "Leverage", leverage, True)
		df.insert(2, "Domain", outside, True)
		return df
	
	def calculate_residuals(df):
		df.insert(2, "Residual", df['Actual']-df['Prediction'], True)
		return df
	def calculate_standard_residuals(df):
		df.insert(2, "Standard Residual", df['Residual']/(df['Residual'].std()), True)
		print(df)
		domain = []
		for i in range(0, df.shape[0]):

			if ((df['Residual'][i]/(df['Residual'].std()) > 1.5 ) | (df['Residual'][i]/(df['Residual'].std()) < -1.5)) & (df['Domain'][i] == 'Valid'):
				domain.append('Valid')
			else:
				domain.append('Invalid')

		del df['Domain']
		df.insert(2, 'Domain', domain, True)
		return df

	train_plot = calculate_leverage(res)
	train_plot = calculate_residuals(train_plot)
	train_plot = calculate_standard_residuals(train_plot)

	test_plot = calculate_leverage(res_valid)
	test_plot = calculate_residuals(test_plot)
	test_plot = calculate_standard_residuals(test_plot)
	fig, ax = plt.subplots()
	ax.scatter(train_plot['Leverage'], train_plot['Residual'], marker='o', c='blue', label = 'Train')
	ax.scatter(test_plot['Leverage'], test_plot['Residual'], marker='o', c='red', label = 'Test')
	ax.axhline(y=1.5, xmin=0, xmax=3.0, color='k')
	ax.axhline(y=-1.5, xmin=0.0, xmax=3.0, color='k')
	ax.axvline(x=0.012, ymin=np.min(train_plot['Residual']) - np.min(train_plot['Residual'] * 0.05), ymax=np.max(train_plot['Residual']) + np.max(train_plot['Residual'] * 0.05), color='k')
	#ax.set_xlim([0, np.max(train_plot['Leverage']) + np.max(train_plot['Leverage']) * 0.05])
	ax.legend()


	try:
		# Create target Directory
		os.mkdir("visualizations")
		print("Visualizations Directory Created ") 
	except FileExistsError:
		print("Visualizations Directory already exists. Skipping creation.")

	fig.savefig('visualizations/nn' + str(iterator) + '.png')
	del(res)
	del(res_valid)
	del(model)
	del(X_Train)
	return time_taken, score_train, score_test, score_valid

def runner_nn_c(input_file_loc, iterator = 0):

	logging.getLogger('tensorflow').disabled = True
	os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
	os.system('export KMP_WARNINGS=FALSE')
	warnings.filterwarnings("ignore", category=FutureWarning)
	dirName = 'pickled'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists. Skipping creation.")
	dirName = 'predictions'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists. Skipping creation.")

	# function to run random forest class
		def run_nn( X_Train , Y_Train, X_Test, Y_Test,  l1_out=512, 
				 l2_out=512, 
				 l1_drop=0.2, 
				 l2_drop=0.2, 
				 l3_out = 512,
				 l3_drop = 0.2,
				 batch_size=100, 
				 epochs=10, cv = 7):
			_nn = nnc.nnc(cv = cv, l1_out = l1_out, l2_out = l2_out, l3_out = l3_out, l1_drop = l1_drop, l2_drop = l2_drop, l3_drop = l3_drop, X_Train = X_Train, Y_Train = Y_Train, X_Test = X_Test, Y_Test = Y_Test)
			nn_evaluation = _nn.nn_evaluate()
			return nn_evaluation

	if input_file_loc:
		with open(input_file_loc, 'r') as f:
			datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]
	if datastore['saved_hyperparameters?']['content'] == "False":
		start = timer()
		
		if datastore['saved_descriptors?']['content'] == "False":
			selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename, chemID = datastore["chemID"]["content"])
			print("-----------------------------------")
			print("Cleaning Data")
			print("-----------------------------------\n")
			inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
			print("-----------------------------------")
			print("Curating Descriptors")
			print("-----------------------------------\n")
			print(f"Number of Compounds: {inDF.shape[0]}")
			inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold =  datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
			activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF, nn_cols = prepare.partition(df = inDF,validset =  datastore['valid_split']['content'], testset = datastore['test_split']['content'], IDboolean = IDboolean)
			print("-----------------------------------")
			print("Partitioning Data")
			print("-----------------------------------\n")
			dfdict = {
			"activityValidDF": activityValidDF,
			"activityTrainDF": activityTrainDF,
			"activityTestDF": activityTestDF,
			"IDValidDF":IDValidDF,
			"IDTrainDF":IDTrainDF,
			"IDTestDF":IDTestDF,
			"validDF":validDF,
			"trainDF":trainDF,
			"testDF":testDF,
			"nameValidDF":nameValidDF,
			"nameTrainDF":nameTrainDF,
			"nameTestDF":nameTestDF,
			"nn_cols":nn_cols
			}
			pickle.dump( dfdict, open( current_folder+ "pickled/nn_descriptors.p", "wb" ) )
		else:
			print("-----------------------------------")
			print("Loading Descriptors")
			print("-----------------------------------\n")
			dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
			activityValidDF = dfdict['activityValidDF']
			activityTrainDF = dfdict['activityTrainDF']
			activityTestDF = dfdict['activityTestDF']
			IDValidDF = dfdict['IDValidDF']
			IDTrainDF = dfdict['IDTrainDF']
			IDTestDF = dfdict['IDTestDF']
			validDF = dfdict['validDF']
			trainDF = dfdict['trainDF']
			testDF = dfdict['testDF']
			nameValidDF = dfdict['nameValidDF']
			nameTrainDF = dfdict['nameTrainDF']
			nameTestDF = dfdict['nameTestDF']

		X_Valid = validDF
		Y_Valid = activityValidDF
		X_Train = trainDF
		Y_Train = activityTrainDF
		X_Test = testDF
		Y_Test = activityTestDF
		orilen = X_Train.shape[1]
		bounds = [
				  {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
				  {'name': 'l2_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
				  {'name':'l3_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
				  {'name': 'l1_out',           'type': 'discrete',    'domain': (orilen/4, orilen/3, orilen/2, orilen, orilen*2, orilen*3,orilen*4)},
				  {'name': 'l2_out',           'type': 'discrete',    'domain': ( orilen/4,orilen/3, orilen/2, orilen, orilen*2, orilen*3,orilen*4)},
				   {'name': 'l3_out',           'type': 'discrete',    'domain': ( orilen/4,orilen/3, orilen/2, orilen, orilen*2,orilen*3, orilen*4)},
				  {'name': 'batch_size',       'type': 'discrete',    'domain': (200,300, 400,500, 600, 700)},
				  {'name': 'epochs',           'type': 'discrete',    'domain': (300, 350, 400, 450, 500, 600, 700)}]
		def f(x):
			evaluation = run_nn(
			l1_drop = float(x[:,0]), 
			l2_drop = float(x[:,1]), 
			l3_drop = float(x[:, 2]),
			l1_out = int(x[:,3]),
			l2_out = int(x[:,4]), 
			l3_out = int(x[:, 5]),
			batch_size = int(x[:,6]), 
			epochs = int(x[:,7]), X_Train = X_Train, Y_Train = Y_Train ,X_Test = X_Test, Y_Test =Y_Test, cv = datastore['cvfolds?']['content'])
			return evaluation
		print("-----------------------------------")
		print("Bayesian Optimization Initiated: First Picking 5 Random Sample Points")
		print("-----------------------------------\n")
		BOModel = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, maximize=True, num_cores = datastore['num_cores']['content'])
		print("-----------------------------------")
		print("Bayesian Optimization: Now Searching 20 Points")
		print("-----------------------------------\n")
		BOModel.run_optimization(max_iter=20)
		print("-----------------------------------")
		print("Bayesian Optimization Converged")
		print("-----------------------------------\n")
		print("-----------------------------------")
		print("Best Hyperparameters Found:\n")
		best_l1_drop = BOModel.x_opt[0]
		best_l2_drop = BOModel.x_opt[1]
		best_l3_drop = BOModel.x_opt[2]
		best_l1_out = BOModel.x_opt[3]
		best_l2_out = BOModel.x_opt[4]
		best_l3_out = BOModel.x_opt[5]
		best_batch_size = BOModel.x_opt[6]
		best_epochs = BOModel.x_opt[7]
		print(f"Layer 1 Drop: {best_l1_drop}")
		print(f"Layer 2 Drop: {best_l2_drop}")
		print(f"Layer 3 Drop: {best_l3_drop}")
		print(f"Layer 1 Neurons: {best_l1_out}")
		print(f"Layer 2 Neurons: {best_l2_out}")
		print(f"Layer 3 Neurons: {best_l3_out}")
		print(f"Batch Size: {best_batch_size}")
		print(f"Epochs: {best_epochs}")
		print("-----------------------------------\n")
		hypdict = {
			"l1_drop":best_l1_drop,
			"l2_drop":best_l2_drop,
			"l3_drop":best_l3_drop,
			"l1_out":best_l1_out,
			"l2_out":best_l2_out,
			"l3_out":best_l3_out,
			"batch_size":best_batch_size,
			"epochs":best_epochs}
		pickle.dump( hypdict, open(current_folder+ "pickled/nn_hyperparameters.p", "wb" ))
	else:
		start = timer()
		print("-----------------------------------")
		print("Loading Descriptors")
		print("-----------------------------------\n")
		dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
		activityValidDF = dfdict['activityValidDF']
		activityTrainDF = dfdict['activityTrainDF']
		activityTestDF = dfdict['activityTestDF']
		IDValidDF = dfdict['IDValidDF']
		IDTrainDF = dfdict['IDTrainDF']
		IDTestDF = dfdict['IDTestDF']
		validDF = dfdict['validDF']
		trainDF = dfdict['trainDF']
		testDF = dfdict['testDF']
		nameValidDF = dfdict['nameValidDF']
		nameTrainDF = dfdict['nameTrainDF']
		nameTestDF = dfdict['nameTestDF']

		X_Valid = validDF
		Y_Valid = activityValidDF
		X_Train = trainDF
		Y_Train = activityTrainDF
		X_Test = testDF
		Y_Test = activityTestDF

		print("-----------------------------------")
		print(f"Loading Optimized Parameters")
		print("-----------------------------------\n")
		hypdict  = pickle.load( open( current_folder + "pickled/nn_hyperparameters.p", "rb" ) )
		best_l1_drop = hypdict["l1_drop"]
		best_l2_drop = hypdict["l2_drop"]
		best_l3_drop = hypdict["l3_drop"]
		best_l1_out = hypdict['l1_out']
		best_l2_out = hypdict['l2_out']
		best_l3_out = hypdict['l3_out']
		best_batch_size = hypdict['batch_size']
		best_epochs = hypdict['epochs']
	print("-----------------------------------")
	print(f"Training Neural Network with Optimized Parameters")
	print("-----------------------------------\n")
	model = Sequential()
	model.add(Dense(int(best_l1_out), input_shape=(X_Train.shape[1], ), kernel_initializer = 'uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(best_l1_drop))
	model.add(Dense(int(best_l2_out), activation='relu',
	                    kernel_initializer = 'uniform'))
	model.add(Dropout(best_l2_drop))
	model.add(Activation('relu'))
	model.add(Dense(int(best_l3_out), activation = 'sigmoid', kernel_initializer = 'uniform'))
	model.add(Dropout(best_l3_drop))
 	        #model.add(Dropout(self.l3_drop))
	model.add(Dense(1,activation = 'sigmoid'))
	model.compile(loss='mean_squared_error',
 	                      optimizer=Adam(), metrics = ['accuracy'])

	early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
			
	model.fit(X_Train, Y_Train,
						   batch_size=int(best_batch_size),
						   epochs=int(best_epochs),
						   verbose=1)

	print("-----------------------------------")
	print(f"Testing Neural Network with Optimized Parameters")
	print("-----------------------------------\n")
	y_pred = np.round(model.predict(X_Test))
	y_pred_train = np.round(model.predict(X_Train))
	y_pred_valid = np.round(model.predict(X_Valid))
	score_test = me.r2_score(Y_Test, y_pred)
	score_train = me.r2_score(Y_Train, y_pred_train)
	score_valid = me.r2_score(Y_Valid, y_pred_valid)

	model.fit(X_Test, Y_Test, batch_size = int(best_batch_size), epochs=int(best_epochs), verbose = 0)

	model.save(current_folder + 'pickled/nnmodel.h5')
	#pickle.dump(nnmodeldict, open("saved/nnmodel.p", "wb"))
	pickle.dump( score_valid, open( current_folder + "pickled/nn_validscore.p", "wb" ) )
	print("-----------------------------------")
	print(f"Final Results")
	print(f"Training R-squared: {score_train}")
	print(f"Testing R-squared: {score_test}")
	print(f"Validation R-squared: {score_valid}")
	print("-----------------------------------\n")
	end = timer()
	time_taken = end - start
	print(f"Time Taken: {time_taken} seconds")
	try:
		os.remove(current_folder + "tmpSDF.sdf")
	except FileNotFoundError:
		print("File Not Found")
	print("-----------------------------------")
	print("Saving Predictions...")
	print("-----------------------------------\n")

	if datastore['chemID']['content'] == 'NA':
		IDboolean = False
	else:
		IDboolean = True

	if IDboolean:
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		NAMESList = []
		for i in range(0,IDTestDF.shape[0]):
			NAMESList.append(nameTestDF.loc[:,].values[i])
			SMILESTest.append(IDTestDF.loc[:,].values[i])
			YTestList.append(Y_Test.loc[:,].values[i])
			YTestPredList.append(y_pred[i][0])
		for i in range(0,IDTrainDF.shape[0]):
			NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i][0])

		res = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i][0])

		res_valid = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
	else:
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		SMILESValid = []
		YValidList = []
		YValidPredList = []
		for i in range(0,IDTestDF.shape[0]):
			SMILESTest.append(IDTestDF.loc[:,].values[i])
			YTestList.append(Y_Test.loc[:,].values[i])
			YTestPredList.append(y_pred[i][0])
		for i in range(0,IDTrainDF.shape[0]):
			#NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i][0])

		res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		#NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			#NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i][0])
			
		res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})


	res.to_csv(current_folder + 'predictions/nn_test.csv', sep=',')
	res_valid.to_csv(current_folder + 'predictions/nn_valid.csv', sep=',')
	print("-----------------------------------")
	print("Neural Network Finished!")
	print("-----------------------------------\n")


	print("-----------------------------------")
	print("Time to do visualizations!")
	print("-----------------------------------\n")

	## df is a dataframe containing the smiles, actual, and prediction
	## returns the dataframe containing leverages
	def calculate_leverage(df):
		actualmean = df['Actual'].mean()
		num = df.shape[0]

		denom = 0
		for i in range(0, num):
			denom += (df['Actual'][i] - actualmean) ** 2.

		outside=[]
		leverage = []
		for i in range(0, num):
			leverage_i = ((df['Actual'][i] - actualmean)** 2.)/(denom)  + (1/num)
			leverage.append(leverage_i)
			if leverage_i > 0.012:
				outside.append('Invalid')
			else:
				outside.append('Valid')
		df.insert(2, "Leverage", leverage, True)
		df.insert(2, "Domain", outside, True)
		return df
	
	def calculate_residuals(df):
		df.insert(2, "Residual", df['Actual']-df['Prediction'], True)
		return df
	def calculate_standard_residuals(df):
		df.insert(2, "Standard Residual", df['Residual']/(df['Residual'].std()), True)
		print(df)
		domain = []
		for i in range(0, df.shape[0]):

			if ((df['Residual'][i]/(df['Residual'].std()) > 1.5 ) | (df['Residual'][i]/(df['Residual'].std()) < -1.5)) & (df['Domain'][i] == 'Valid'):
				domain.append('Valid')
			else:
				domain.append('Invalid')

		del df['Domain']
		df.insert(2, 'Domain', domain, True)
		return df

	train_plot = calculate_leverage(res)
	train_plot = calculate_residuals(train_plot)
	train_plot = calculate_standard_residuals(train_plot)

	test_plot = calculate_leverage(res_valid)
	test_plot = calculate_residuals(test_plot)
	test_plot = calculate_standard_residuals(test_plot)
	fig, ax = plt.subplots()
	ax.scatter(train_plot['Leverage'], train_plot['Residual'], marker='o', c='blue', label = 'Train')
	ax.scatter(test_plot['Leverage'], test_plot['Residual'], marker='o', c='red', label = 'Test')
	ax.axhline(y=1.5, xmin=0, xmax=3.0, color='k')
	ax.axhline(y=-1.5, xmin=0.0, xmax=3.0, color='k')
	ax.axvline(x=0.012, ymin=np.min(train_plot['Residual']) - np.min(train_plot['Residual'] * 0.05), ymax=np.max(train_plot['Residual']) + np.max(train_plot['Residual'] * 0.05), color='k')
	#ax.set_xlim([0, np.max(train_plot['Leverage']) + np.max(train_plot['Leverage']) * 0.05])
	ax.legend()


	try:
		# Create target Directory
		os.mkdir("visualizations")
		print("Visualizations Directory Created ") 
	except FileExistsError:
		print("Visualizations Directory already exists. Skipping creation.")

	fig.savefig('visualizations/nn' + str(iterator) + '.png')
	del(res)
	del(res_valid)
	del(model)
	del(X_Train)
	return time_taken, score_train, score_test, score_valid
# def runner_nn_c(input_file_loc):

# 	logging.getLogger('tensorflow').disabled = True
# 	os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
# 	os.system('export KMP_WARNINGS=FALSE')
# 	warnings.filterwarnings("ignore", category=FutureWarning)
# 	dirName = 'pickled'
# 	try:
# 		# Create target Directory
# 		os.mkdir(dirName)
# 		print("Directory " , dirName ,  " Created ") 
# 	except FileExistsError:
# 		print("Directory " , dirName ,  " already exists. Skipping creation.")
# 	dirName = 'predictions'
# 	try:
# 		# Create target Directory
# 		os.mkdir(dirName)
# 		print("Directory " , dirName ,  " Created ") 
# 	except FileExistsError:
# 		print("Directory " , dirName ,  " already exists. Skipping creation.")

# 	# function to run random forest class
# 	def run_nn( X_Train , Y_Train, X_Test, Y_Test,  l1_out=512, 
# 	                 l2_out=512, 
# 	                 l1_drop=0.2, 
# 	                 l2_drop=0.2, 
# 	                 l3_out = 512,
# 	                 l3_drop = 0.2,
# 	                 batch_size=100, 
# 	                 epochs=10, cv = 7):
# 	    _nn = nnc.nnc(cv = cv, l1_out = l1_out, l2_out = l2_out, l3_out = l3_out, l1_drop = l1_drop, l2_drop = l2_drop, l3_drop = l3_drop, X_Train = X_Train, Y_Train = Y_Train, X_Test = X_Test, Y_Test = Y_Test)
# 	    nn_evaluation = _nn.nn_evaluate()
# 	    return nn_evaluation

# 	if input_file_loc:
# 	    with open(input_file_loc, 'r') as f:
# 	        datastore = json.load(f)
# 	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
# 	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]
# 	if datastore['saved_hyperparameters?']['content'] == "False":
# 		start = timer()
		
# 		if datastore['saved_descriptors?']['content'] == "False":
# 			selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename, chemID = datastore["chemID"]["content"])
# 			print("-----------------------------------")
# 			print("Cleaning Data")
# 			print("-----------------------------------\n")
# 			inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
# 			print("-----------------------------------")
# 			print("Curating Descriptors")
# 			print("-----------------------------------\n")
# 			print(f"Number of Compounds: {inDF.shape[0]}")
# 			inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold =  datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
# 			activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF = prepare.partition(df = inDF,validset =  datastore['valid_split']['content'], testset = datastore['test_split']['content'], IDboolean = IDboolean)
# 			print("-----------------------------------")
# 			print("Partitioning Data")
# 			print("-----------------------------------\n")
# 			dfdict = {
# 			"activityValidDF": activityValidDF,
# 			"activityTrainDF": activityTrainDF,
# 			"activityTestDF": activityTestDF,
# 			"IDValidDF":IDValidDF,
# 			"IDTrainDF":IDTrainDF,
# 			"IDTestDF":IDTestDF,
# 			"validDF":validDF,
# 			"trainDF":trainDF,
# 			"testDF":testDF,
# 			"nameValidDF":nameValidDF,
# 			"nameTrainDF":nameTrainDF,
# 			"nameTestDF":nameTestDF
# 			}
# 			pickle.dump( dfdict, open( current_folder+ "pickled/nn_descriptors.p", "wb" ) )
# 		else:
# 			print("-----------------------------------")
# 			print("Loading Descriptors")
# 			print("-----------------------------------\n")
# 			dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
# 			activityValidDF = dfdict['activityValidDF']
# 			activityTrainDF = dfdict['activityTrainDF']
# 			activityTestDF = dfdict['activityTestDF']
# 			IDValidDF = dfdict['IDValidDF']
# 			IDTrainDF = dfdict['IDTrainDF']
# 			IDTestDF = dfdict['IDTestDF']
# 			validDF = dfdict['validDF']
# 			trainDF = dfdict['trainDF']
# 			testDF = dfdict['testDF']
# 			nameValidDF = dfdict['nameValidDF']
# 			nameTrainDF = dfdict['nameTrainDF']
# 			nameTestDF = dfdict['nameTestDF']

# 		X_Valid = validDF
# 		Y_Valid = activityValidDF
# 		X_Train = trainDF
# 		Y_Train = activityTrainDF
# 		X_Test = testDF
# 		Y_Test = activityTestDF
# 		orilen = X_Train.shape[1]
# 		bounds = [
# 		          {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
# 		          {'name': 'l2_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
# 		          {'name':'l3_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
# 		          {'name': 'l1_out',           'type': 'discrete',    'domain': (orilen/4, orilen/3, orilen/2, orilen, orilen*2, orilen*3,orilen*4)},
# 		          {'name': 'l2_out',           'type': 'discrete',    'domain': ( orilen/4,orilen/3, orilen/2, orilen, orilen*2, orilen*3,orilen*4)},
# 		           {'name': 'l3_out',           'type': 'discrete',    'domain': ( orilen/4,orilen/3, orilen/2, orilen, orilen*2,orilen*3, orilen*4)},
# 		          {'name': 'batch_size',       'type': 'discrete',    'domain': (200,300, 400,500, 600, 700)},
# 		          {'name': 'epochs',           'type': 'discrete',    'domain': (300, 350, 400, 450, 500, 600, 700)}]
# 		def f(x):
# 			evaluation = run_nn(
# 	        l1_drop = float(x[:,0]), 
# 	        l2_drop = float(x[:,1]), 
# 	        l3_drop = float(x[:, 2]),
# 	        l1_out = int(x[:,3]),
# 	        l2_out = int(x[:,4]), 
# 	    	l3_out = int(x[:, 5]),
# 	    	batch_size = int(x[:,6]), 
# 	    	epochs = int(x[:,7]), X_Train = X_Train, Y_Train = Y_Train ,X_Test = X_Test, Y_Test =Y_Test, cv = datastore['cvfolds?']['content'])
# 			return evaluation
# 		print("-----------------------------------")
# 		print("Bayesian Optimization Initiated: First Picking 5 Random Sample Points")
# 		print("-----------------------------------\n")
# 		BOModel = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, maximize=True, num_cores = datastore['num_cores']['content'])
# 		print("-----------------------------------")
# 		print("Bayesian Optimization: Now Searching 20 Points")
# 		print("-----------------------------------\n")
# 		BOModel.run_optimization(max_iter=20)
# 		print("-----------------------------------")
# 		print("Bayesian Optimization Converged")
# 		print("-----------------------------------\n")
# 		print("-----------------------------------")
# 		print("Best Hyperparameters Found:\n")
# 		best_l1_drop = BOModel.x_opt[0]
# 		best_l2_drop = BOModel.x_opt[1]
# 		best_l3_drop = BOModel.x_opt[2]
# 		best_l1_out = BOModel.x_opt[3]
# 		best_l2_out = BOModel.x_opt[4]
# 		best_l3_out = BOModel.x_opt[5]
# 		best_batch_size = BOModel.x_opt[6]
# 		best_epochs = BOModel.x_opt[7]
# 		print(f"Layer 1 Drop: {best_l1_drop}")
# 		print(f"Layer 2 Drop: {best_l2_drop}")
# 		print(f"Layer 3 Drop: {best_l3_drop}")
# 		print(f"Layer 1 Neurons: {best_l1_out}")
# 		print(f"Layer 2 Neurons: {best_l2_out}")
# 		print(f"Layer 3 Neurons: {best_l3_out}")
# 		print(f"Batch Size: {best_batch_size}")
# 		print(f"Epochs: {best_epochs}")
# 		print("-----------------------------------\n")
# 		hypdict = {
# 			"l1_drop":best_l1_drop,
# 			"l2_drop":best_l2_drop,
# 			"l3_drop":best_l3_drop,
# 			"l1_out":best_l1_out,
# 			"l2_out":best_l2_out,
# 			"l3_out":best_l3_out,
# 			"batch_size":best_batch_size,
# 			"epochs":best_epochs}
# 		pickle.dump( hypdict, open(current_folder+ "pickled/nn_hyperparameters.p", "wb" ))
# 	else:
# 		start = timer()
# 		print("-----------------------------------")
# 		print("Loading Descriptors")
# 		print("-----------------------------------\n")
# 		dfdict  = pickle.load( open( current_folder + "pickled/nn_descriptors.p", "rb" ) )
# 		activityValidDF = dfdict['activityValidDF']
# 		activityTrainDF = dfdict['activityTrainDF']
# 		activityTestDF = dfdict['activityTestDF']
# 		IDValidDF = dfdict['IDValidDF']
# 		IDTrainDF = dfdict['IDTrainDF']
# 		IDTestDF = dfdict['IDTestDF']
# 		validDF = dfdict['validDF']
# 		trainDF = dfdict['trainDF']
# 		testDF = dfdict['testDF']
# 		nameValidDF = dfdict['nameValidDF']
# 		nameTrainDF = dfdict['nameTrainDF']
# 		nameTestDF = dfdict['nameTestDF']

# 		X_Valid = validDF
# 		Y_Valid = activityValidDF
# 		X_Train = trainDF
# 		Y_Train = activityTrainDF
# 		X_Test = testDF
# 		Y_Test = activityTestDF

# 		print("-----------------------------------")
# 		print(f"Loading Optimized Parameters")
# 		print("-----------------------------------\n")
# 		hypdict  = pickle.load( open( current_folder + "pickled/nn_hyperparameters.p", "rb" ) )
# 		best_l1_drop = hypdict["l1_drop"]
# 		best_l2_drop = hypdict["l2_drop"]
# 		best_l3_drop = hypdict["l3_drop"]
# 		best_l1_out = hypdict['l1_out']
# 		best_l2_out = hypdict['l2_out']
# 		best_l3_out = hypdict['l3_out']
# 		best_batch_size = hypdict['batch_size']
# 		best_epochs = hypdict['epochs']
# 	print("-----------------------------------")
# 	print(f"Training Neural Network with Optimized Parameters")
# 	print("-----------------------------------\n")
# 	model = Sequential()
# 	model.add(Dense(int(best_l1_out), input_shape=(X_Train.shape[1], ), kernel_initializer = 'uniform'))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(best_l1_drop))
# 	model.add(Dense(int(best_l2_out), activation='relu',
# 	                    kernel_initializer = 'uniform'))
# 	model.add(Dropout(best_l2_drop))
# 	model.add(Activation('relu'))

# 	#model.add(Dense(int(best_l3_out), activation = 'sigmoid', kernel_initializer = 'uniform'))
# 	#model.add(Dropout(best_l3_drop))
# 	        #model.add(Dropout(self.l3_drop))
# 	model.add(Dense(1,activation = 'sigmoid'))
# 	model.compile(loss='mean_squared_error',
# 	                      optimizer=Adam(), metrics = ['accuracy'])

# 	#early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
			
# 	model.fit(X_Train, Y_Train,
# 	                       batch_size=int(best_batch_size),
# 	                       epochs=int(best_epochs),
# 	                       verbose=1)

# 	print("-----------------------------------")
# 	print(f"Testing Neural Network with Optimized Parameters")
# 	print("-----------------------------------\n")
# 	y_pred = model.predict(X_Test)
# 	y_pred_train = model.predict(X_Train)
# 	y_pred_valid = model.predict(X_Valid)
# 	score_test = me.r2_score(Y_Test, y_pred)
# 	score_train = me.r2_score(Y_Train, y_pred_train)
# 	score_valid = me.r2_score(Y_Valid, y_pred_valid)

# 	model.fit(X_Test, Y_Test, batch_size = int(best_batch_size), epochs=int(best_epochs), verbose = 0)

# 	model.save(current_folder + 'pickled/nnmodel.h5')
# 	#pickle.dump(nnmodeldict, open("saved/nnmodel.p", "wb"))
# 	pickle.dump( score_valid, open( current_folder + "pickled/nn_validscore.p", "wb" ) )
# 	print("-----------------------------------")
# 	print(f"Final Results")
# 	print(f"Training R-squared: {score_train}")
# 	print(f"Testing R-squared: {score_test}")
# 	print(f"Validation R-squared: {score_valid}")
# 	print("-----------------------------------\n")
# 	end = timer()
# 	time_taken = end - start
# 	print(f"Time Taken: {time_taken} seconds")
# 	try:
# 		os.remove(current_folder + "tmpSDF.sdf")
# 	except FileNotFoundError:
# 		print("File Not Found")
# 	print("-----------------------------------")
# 	print("Saving Predictions...")
# 	print("-----------------------------------\n")

# 	if datastore['chemID']['content'] == 'NA':
# 		IDboolean = False
# 	else:
# 		IDboolean = True

# 	if IDboolean:
# 		SMILESTest = []
# 		YTestList = []
# 		YTestPredList = []
# 		NAMESList = []
# 		for i in range(0,IDTestDF.shape[0]):
# 			NAMESList.append(nameTestDF.loc[:,].values[i])
# 			SMILESTest.append(IDTestDF.loc[:,].values[i])
# 			YTestList.append(Y_Test.loc[:,].values[i])
# 			YTestPredList.append(y_pred[i][0])
# 		for i in range(0,IDTrainDF.shape[0]):
# 			NAMESList.append(nameTrainDF.loc[:, ].values[i])
# 			SMILESTest.append(IDTrainDF.loc[:,].values[i])
# 			YTestList.append(Y_Train.loc[:,].values[i])
# 			YTestPredList.append(y_pred_train[i][0])

# 		res = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
# 		SMILESTest = []
# 		YTestList = []
# 		YTestPredList = []
# 		NAMESList = []
# 		for i in range(0,IDValidDF.shape[0]):
# 			NAMESList.append(nameValidDF.loc[:, ].values[i])
# 			SMILESTest.append(IDValidDF.loc[:,].values[i])
# 			YTestList.append(Y_Valid.loc[:,].values[i])
# 			YTestPredList.append(y_pred_valid[i][0])

# 		res_valid = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
# 	else:
# 		SMILESTest = []
# 		YTestList = []
# 		YTestPredList = []
# 		SMILESValid = []
# 		YValidList = []
# 		YValidPredList = []
# 		for i in range(0,IDTestDF.shape[0]):
# 			SMILESTest.append(IDTestDF.loc[:,].values[i])
# 			YTestList.append(Y_Test.loc[:,].values[i])
# 			YTestPredList.append(y_pred[i][0])
# 		for i in range(0,IDTrainDF.shape[0]):
# 			#NAMESList.append(nameTrainDF.loc[:, ].values[i])
# 			SMILESTest.append(IDTrainDF.loc[:,].values[i])
# 			YTestList.append(Y_Train.loc[:,].values[i])
# 			YTestPredList.append(y_pred_train[i][0])

# 		res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
# 		SMILESTest = []
# 		YTestList = []
# 		YTestPredList = []
# 		#NAMESList = []
# 		for i in range(0,IDValidDF.shape[0]):
# 			#NAMESList.append(nameValidDF.loc[:, ].values[i])
# 			SMILESTest.append(IDValidDF.loc[:,].values[i])
# 			YTestList.append(Y_Valid.loc[:,].values[i])
# 			YTestPredList.append(y_pred_valid[i][0])
			
# 		res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})


# 	res.to_csv(current_folder + 'predictions/nn_test.csv', sep=',')
# 	res_valid.to_csv(current_folder + 'predictions/nn_valid.csv', sep=',')
# 	print("-----------------------------------")
# 	print("Neural Network Finished!")
# 	print("-----------------------------------\n")
# 	del(res)
# 	del(res_valid)
# 	del(model)
# 	del(X_Train)
# 	return time_taken, score_train, score_test, score_valid
