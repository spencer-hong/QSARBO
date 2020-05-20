import GPy, GPyOpt
import numpy as np
import pandas as pd
import sklearn.model_selection as mose
import sklearn.preprocessing as skp
import sklearn.ensemble as sken
import sklearn.neighbors as skne
import sklearn.metrics as me 
from timeit import default_timer as timer
from padelpy import from_smiles
import matplotlib.pyplot as plt 
from padelpy import from_smiles

import json
import pickle
#from datetime import datetime
import random

import sys, os

sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classes import prepare as prepare 
from models.classes import randomforest as randomforest 
from models.classes import randomforestc as randomforestc
del sys.path[0]

#random.seed(36)

def runner_rf(input_file_loc, iterator = 0):
	# function to run random forest class
	def run_rf( X_Train , Y_Train, X_Test, Y_Test, n_estimators = 100, max_features = 0.5, max_depth = 0.3, min_samples_split = 2 ,cv = 7):
	    _randomforest = randomforest.randomforest(cv = cv, n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, X_Train = X_Train, Y_Train = Y_Train, X_Test = X_Test, Y_Test = Y_Test)
	    rf_evaluation = _randomforest.rf_evaluate()
	    return rf_evaluation

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
	if input_file_loc:
	    with open(input_file_loc, 'r') as f:
	        datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]

	if datastore['saved_hyperparameters?']['content'] == "False":
		start = timer()
		if datastore['saved_descriptors?']['content'] == "False":
			selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename , chemID = datastore["chemID"]["content"])
			print("-----------------------------------")
			print("Cleaning Data")
			print("-----------------------------------\n")
			inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
			print("-----------------------------------")
			print("Curating Descriptors")
			print("-----------------------------------\n")
			print(f"Number of Compounds: {inDF.shape[0]}")
			ds = []
			print(inDF.shape[0])
			print(inDF.shape[1])



			#outDF = pd.DataFrame(ds)
			with open('padel.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				outDF = pickle.load(f)

			cols = outDF.columns.tolist()
			cols = cols[-2:] + cols[:-2]
			outDF = outDF[cols]
			outDF_1 = outDF.dropna()

			#outDF_1.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()

			#print(outDF_1)

			#print('done')
			#print(result)
			#result = pd.concat([inDF['logki'], outDF], sort = False)

			#result_new = result._get_numeric_data()
			#print(result_new)
			#inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold =  datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
			#print(inDF.head)

			activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF = prepare.partition(df = outDF_1,validset =  datastore['valid_split']['content'], testset = datastore['test_split']['content'], IDboolean = IDboolean)
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
			"nameTestDF":nameTestDF
			}
			pickle.dump( dfdict, open( current_folder + "pickled/rf_descriptors.p", "wb" ) )
		else:
			print("-----------------------------------")
			print("Loading Descriptors")
			print("-----------------------------------\n")
			dfdict  = pickle.load( open( current_folder + "rf_descriptors.p", "rb" ) )
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

		bounds = [
		          {'name': 'n_estimators',          'type': 'continuous',  'domain': (200, 400)},
		          {'name': 'max_features',          'type': 'continuous',  'domain': (0.01, 0.99)},
		          {'name':'max_depth', 'type': 'continuous', 'domain': (1, 25)},
		          {'name': 'min_samples_split',           'type': 'continuous',    'domain': (2, 15)}]

		def f(x):
			#print(x)
			evaluation = run_rf(
		        n_estimators = float(x[:,0]), 
		        max_features = float(x[:,1]), 
		        max_depth = float(x[:, 2]),
		        min_samples_split = int(x[:,3]), X_Train = X_Train, Y_Train = Y_Train ,X_Test = X_Test, Y_Test =Y_Test, cv=datastore['cvfolds?']['content'])
			return evaluation
		start = timer()
		print("-----------------------------------")
		print("Bayesian Optimization Initiated: First Picking 5 Random Sample Points")
		print("-----------------------------------\n")
		BOModel = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, maximize = True,num_cores = datastore['num_cores']['content'])
		print("-----------------------------------")
		print("Bayesian Optimization: Now Searching 20 Points")
		print("-----------------------------------\n")
		BOModel.run_optimization(max_iter=20)
		print("-----------------------------------")
		print("Bayesian Optimization Converged")
		print("-----------------------------------\n")
		print("-----------------------------------")
		print("Best Hyperparameters Found:\n")
		best_n_estimators = BOModel.x_opt[0]
		best_max_features = BOModel.x_opt[1]
		best_max_depth = BOModel.x_opt[2]
		best_min_samples_split = BOModel.x_opt[3]
		print(f"# of Estimators: {best_n_estimators}")
		print(f"Max Features: {best_max_features}")
		print(f"Max Depth: {best_max_depth}")
		print(f"Min Samples Split: {best_min_samples_split}")
		print("-----------------------------------\n")
		hypdict = {
			"n_estimators":best_n_estimators,
			"max_features":best_max_features,
			"max_depth":best_max_depth,
			"min_samples_split":best_min_samples_split
			}
		pickle.dump( hypdict, open( current_folder + "pickled/rf_hyperparameters.p", "wb" ) )
	else:
		start = timer()
		print("-----------------------------------")
		print("Loading Descriptors")
		print("-----------------------------------\n")
		dfdict  = pickle.load( open( current_folder + "pickled/rf_descriptors.p", "rb" ) )
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
		hypdict  = pickle.load( open( current_folder + "pickled/rf_hyperparameters.p", "rb" ) )
		best_n_estimators = hypdict['n_estimators']
		best_max_features = hypdict['max_features']
		best_max_depth = hypdict['max_depth']
		best_min_samples_split = hypdict['min_samples_split']

	print("-----------------------------------")
	print(f"Training Random Forest with Optimized Parameters")
	print("-----------------------------------\n")
	rfmodel = sken.RandomForestRegressor(min_samples_split = int(best_min_samples_split),  max_depth = best_max_depth, n_estimators = int(best_n_estimators), max_features = best_max_features, verbose=1, n_jobs = -1)

	rfmodel.fit(X_Train, Y_Train)
	print("-----------------------------------")
	print(f"Testing Random Forest with Optimized Parameters")
	print("-----------------------------------\n")
	y_pred = rfmodel.predict(X_Test)
	y_pred_train = rfmodel.predict(X_Train)
	y_pred_valid = rfmodel.predict(X_Valid)
	score_test = me.r2_score(Y_Test, y_pred)
	score_train = me.r2_score(Y_Train, y_pred_train)
	score_valid = me.r2_score(Y_Valid, y_pred_valid)

	pickle.dump( score_valid, open( current_folder + "pickled/rf_validscore.p", "wb" ) )
	pickle.dump(score_test, open("pickled/rf_testscore.p", "wb"))

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
	#SMILESColTest = IDTestDF[:, 0]
	#print(SMILESColTest)
	#print(type(SMILESColTest))
	#SMILESColValid = IDValidDF[:, 0].tolist()
	#Y_Test = Y_Test[:, 0].tolist()
	#Y_Valid = Y_Valid[:, 0].tolist()
	#res = pd.DataFrame(columns=('SMILES', 'Actual', 'Prediction'))
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
			YTestPredList.append(y_pred[i])
		for i in range(0,IDTrainDF.shape[0]):
			NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i])

		res = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i])

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
			YTestPredList.append(y_pred[i])
		for i in range(0,IDTrainDF.shape[0]):
			#NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i])

		res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		#NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			#NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i])
			
		res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})

	res.to_csv(current_folder + 'predictions/rf_test.csv', sep=',')
	res_valid.to_csv(current_folder + 'predictions/rf_valid.csv', sep=',')
	print("-----------------------------------")
	print("Random Forest Finished!")
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
		domain = []
		if (df['Residual']/(df['Residual'].std()) > 1.5 or df['Residual']/(df['Residual'].std()) < -1.5) and df['Domain'] == 'Valid':
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

	fig.savefig('visualizations/rf' + iterator +'.png')

	del(res)
	del(res_valid)
	del(rfmodel)
	del(X_Train)
	return time_taken, score_train, score_test, score_valid



def runner_rf_c(input_file_loc):
	# function to run random forest class
	def run_rf( X_Train , Y_Train, X_Test, Y_Test, n_estimators = 100, max_features = 0.5, max_depth = 0.3, min_samples_split = 2 ,cv = 7):
	    _randomforest = randomforestc.randomforestc(cv = cv, n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, X_Train = X_Train, Y_Train = Y_Train, X_Test = X_Test, Y_Test = Y_Test)
	    rf_evaluation = _randomforest.rf_evaluate()
	    return rf_evaluation

	dirName = 'pickled'
	start_time = timer()
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
	if input_file_loc:
	    with open(input_file_loc, 'r') as f:
	        datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]

	if datastore['saved_hyperparameters?']['content'] == "False":
		start = timer()
		if datastore['saved_descriptors?']['content'] == "False":
			selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename , chemID = datastore["chemID"]["content"])
			print("-----------------------------------")
			print("Cleaning Data")
			print("-----------------------------------\n")
			inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
			print("-----------------------------------")
			print("Curating Descriptors")
			print("-----------------------------------\n")
			print(f"Number of Compounds: {inDF.shape[0]}")
			inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold =  datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
			#print(inDF.head)
			activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF = prepare.partition(df = inDF,validset =  datastore['valid_split']['content'], testset = datastore['test_split']['content'], IDboolean = IDboolean)
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
			"nameTestDF":nameTestDF
			}
			pickle.dump( dfdict, open( current_folder + "pickled/rf_descriptors.p", "wb" ) )
		else:
			print("-----------------------------------")
			print("Loading Descriptors")
			print("-----------------------------------\n")
			dfdict  = pickle.load( open( current_folder + "pickled/rf_descriptors.p", "rb" ) )
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

		bounds = [
		          {'name': 'n_estimators',          'type': 'continuous',  'domain': (200, 400)},
		          {'name': 'max_features',          'type': 'continuous',  'domain': (0.01, 0.99)},
		          {'name':'max_depth', 'type': 'continuous', 'domain': (1, 18)},
		          {'name': 'min_samples_split',           'type': 'continuous',    'domain': (2, 15)}]

		def f(x):
			#print(x)
			evaluation = run_rf(
		        n_estimators = float(x[:,0]), 
		        max_features = float(x[:,1]), 
		        max_depth = float(x[:, 2]),
		        min_samples_split = int(x[:,3]), X_Train = X_Train, Y_Train = Y_Train ,X_Test = X_Test, Y_Test =Y_Test, cv=datastore['cvfolds?']['content'])
			return evaluation
		print("-----------------------------------")
		print("Bayesian Optimization Initiated: First Picking 5 Random Sample Points")
		print("-----------------------------------\n")
		BOModel = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, maximize = True,num_cores = datastore['num_cores']['content'])
		print("-----------------------------------")
		print("Bayesian Optimization: Now Searching 20 Points")
		print("-----------------------------------\n")
		BOModel.run_optimization(max_iter=20)
		print("-----------------------------------")
		print("Bayesian Optimization Converged")
		print("-----------------------------------\n")
		print("-----------------------------------")
		print("Best Hyperparameters Found:\n")
		best_n_estimators = BOModel.x_opt[0]
		best_max_features = BOModel.x_opt[1]
		best_max_depth = BOModel.x_opt[2]
		best_min_samples_split = BOModel.x_opt[3]
		print(f"# of Estimators: {best_n_estimators}")
		print(f"Max Features: {best_max_features}")
		print(f"Max Depth: {best_max_depth}")
		print(f"Min Samples Split: {best_min_samples_split}")
		print("-----------------------------------\n")
		hypdict = {
			"n_estimators":best_n_estimators,
			"max_features":best_max_features,
			"max_depth":best_max_depth,
			"min_samples_split":best_min_samples_split
			}
		pickle.dump( hypdict, open( current_folder + "pickled/rf_hyperparameters.p", "wb" ) )
	else:
		start = timer()
		print("-----------------------------------")
		print("Loading Descriptors")
		print("-----------------------------------\n")
		dfdict  = pickle.load( open( current_folder + "pickled/rf_descriptors.p", "rb" ) )
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
		hypdict  = pickle.load( open( current_folder + "pickled/rf_hyperparameters.p", "rb" ) )
		best_n_estimators = hypdict['n_estimators']
		best_max_features = hypdict['max_features']
		best_max_depth = hypdict['max_depth']
		best_min_samples_split = hypdict['min_samples_split']

	print("-----------------------------------")
	print(f"Training Random Forest with Optimized Parameters")
	print("-----------------------------------\n")
	rfmodel = sken.RandomForestClassifier(min_samples_split = int(best_min_samples_split),  max_depth = best_max_depth, n_estimators = int(best_n_estimators), max_features = best_max_features, verbose=1, n_jobs = -1)

	rfmodel.fit(X_Train, Y_Train)
	print("-----------------------------------")
	print(f"Testing Random Forest with Optimized Parameters")
	print("-----------------------------------\n")
	y_pred = rfmodel.predict(X_Test)
	y_pred_train = rfmodel.predict(X_Train)
	y_pred_valid = rfmodel.predict(X_Valid)
	score_test = me.r2_score(Y_Test, y_pred)
	score_train = me.r2_score(Y_Train, y_pred_train)
	score_valid = me.r2_score(Y_Valid, y_pred_valid)

	pickle.dump( score_valid, open( current_folder + "pickled/rf_validscore.p", "wb" ) )
	pickle.dump(score_test, open("pickled/rf_testscore.p", "wb"))

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
	#SMILESColTest = IDTestDF[:, 0]
	#print(SMILESColTest)
	#print(type(SMILESColTest))
	#SMILESColValid = IDValidDF[:, 0].tolist()
	#Y_Test = Y_Test[:, 0].tolist()
	#Y_Valid = Y_Valid[:, 0].tolist()
	#res = pd.DataFrame(columns=('SMILES', 'Actual', 'Prediction'))
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
			YTestPredList.append(y_pred[i])
		for i in range(0,IDTrainDF.shape[0]):
			NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i])

		res = pd.DataFrame({'SMILES':SMILESTest, 'Chemical ID': NAMESList, 'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i])

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
			YTestPredList.append(y_pred[i])
		for i in range(0,IDTrainDF.shape[0]):
			#NAMESList.append(nameTrainDF.loc[:, ].values[i])
			SMILESTest.append(IDTrainDF.loc[:,].values[i])
			YTestList.append(Y_Train.loc[:,].values[i])
			YTestPredList.append(y_pred_train[i])

		res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
		SMILESTest = []
		YTestList = []
		YTestPredList = []
		#NAMESList = []
		for i in range(0,IDValidDF.shape[0]):
			#NAMESList.append(nameValidDF.loc[:, ].values[i])
			SMILESTest.append(IDValidDF.loc[:,].values[i])
			YTestList.append(Y_Valid.loc[:,].values[i])
			YTestPredList.append(y_pred_valid[i])
			
		res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})

	res.to_csv(current_folder + 'predictions/rf_test.csv', sep=',')
	res_valid.to_csv(current_folder + 'predictions/rf_valid.csv', sep=',')
	print("-----------------------------------")
	print("Random Forest Finished!")
	print("-----------------------------------\n")







	del(res)
	del(res_valid)
	del(rfmodel)
	del(X_Train)
	return time_taken, score_train, score_test, score_valid
