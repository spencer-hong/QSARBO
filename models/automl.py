from tpot import TPOTRegressor
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from models.classes import prepare as prepare
from models.classes import randomforest as randomforest
from models.classes import randomforestc as randomforestc
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import random, os, json, datetime
from timeit import default_timer as timer

#random.seed(36)

def tpot_c(input_file_loc):
	dirName = 'pickled'
	try:
	# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName , " Created ")
	except FileExistsError:
		print("Directory " , dirName , " already exists. Skipping creation.")

	dirName = 'predictions'
	try:
	# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName , " Created ")
	except FileExistsError:
		print("Directory " , dirName , " already exists. Skipping creation.")
	if input_file_loc:
		with open(input_file_loc, 'r') as f:
			datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]
	selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename , chemID = datastore["chemID"]["content"])
	print("-----------------------------------")

	print("Cleaning Data")
	print("-----------------------------------\n")
	inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
	print("-----------------------------------")
	print("Curating Descriptors")
	print("-----------------------------------\n")
	print(f"Number of Compounds: {inDF.shape[0]}")
	inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold = datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
	#print(inDF.head)
	activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF, _= prepare.partition(df = inDF,validset = datastore['valid_split']['content'], testset = datastore['test_split'] ['content'], IDboolean = IDboolean)
	print("-----------------------------------")
	print("Partitioning Data")
	print("-----------------------------------\n")

	X_Valid = validDF
	Y_Valid = activityValidDF
	X_Train = trainDF
	Y_Train = activityTrainDF
	X_Test = testDF
	Y_Test = activityTestDF

	#print(X_Valid)
	#print(Y_Train)
	# Make a custom metric function
	def my_custom_accuracy(y_true, y_pred):
		return r2_score(y_true, y_pred)
	my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)
	start = timer()
	tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, cv = 10, n_jobs = -1, use_dask = False, periodic_checkpoint_folder = '/Users/spencerhong/Documents/QSARBayesOpt/autotest/tpot_check')
	tpot.fit(X_Train, Y_Train)
	Y_Test_Pred = tpot.predict(X_Test)
	Y_Train_Pred = tpot.predict(X_Train)
	Y_Valid_Pred = tpot.predict(X_Valid)

	SMILESTest = []
	YTestList = []
	YTestPredList = []
	SMILESValid = []
	YValidList = []
	YValidPredList = []
	for i in range(0,IDTestDF.shape[0]):
		SMILESTest.append(IDTestDF.loc[:,].values[i])
		YTestList.append(Y_Test.loc[:,].values[i])
		YTestPredList.append(Y_Test_Pred[i])
	for i in range(0,IDTrainDF.shape[0]):
		#NAMESList.append(nameTrainDF.loc[:, ].values[i])
		SMILESTest.append(IDTrainDF.loc[:,].values[i])
		YTestList.append(Y_Train.loc[:,].values[i])
		YTestPredList.append(Y_Train_Pred[i])

	res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
	SMILESTest = []
	YTestList = []
	YTestPredList = []
	#NAMESList = []
	for i in range(0,IDValidDF.shape[0]):
		#NAMESList.append(nameValidDF.loc[:, ].values[i])
		SMILESTest.append(IDValidDF.loc[:,].values[i])
		YTestList.append(Y_Valid.loc[:,].values[i])
		YTestPredList.append(Y_Valid_Pred[i])
		
	res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
	res.to_csv(current_folder + 'predictions/automl_test.csv', sep=',')
	res_valid.to_csv(current_folder + 'predictions/automl_valid.csv', sep=',')
	print(r2_score(Y_Test, Y_Test_Pred))
	print('---------------------------\n')
	print('TIME')
	end = timer()
	time_duration = end - start
	print(f"Time taken: {time_duration}") # Time in seconds, e.g. 5.38091952400282
	tpot.export('tpot_classification.py')

	del(res)
	del(res_valid)
	del(X_Train)

	return time_duration, r2_score(Y_Train, Y_Train_Pred), r2_score(Y_Test, Y_Test_Pred), r2_score(Y_Valid, Y_Valid_Pred)

def tpot_r(input_file_loc):
	dirName = 'pickled'
	try:
	# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName , " Created ")
	except FileExistsError:
		print("Directory " , dirName , " already exists. Skipping creation.")

	dirName = 'predictions'
	try:
	# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName , " Created ")
	except FileExistsError:
		print("Directory " , dirName , " already exists. Skipping creation.")
	if input_file_loc:
		with open(input_file_loc, 'r') as f:
			datastore = json.load(f)
	current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
	filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/' +datastore["dataset_name"]["content"]
	selected_data, IDboolean = prepare.isolate(structname= datastore["column_SMILES"]['content'], activityname = datastore["column_activity"]["content"], filelocation = filename , chemID = datastore["chemID"]["content"])
	print("-----------------------------------")

	print("Cleaning Data")
	print("-----------------------------------\n")
	inDF = prepare.cleanSMILES(df = selected_data, elementskept = datastore["elements_kept"]["content"], smilesName = datastore["column_SMILES"]["content"])
	print("-----------------------------------")
	print("Curating Descriptors")
	print("-----------------------------------\n")
	print(f"Number of Compounds: {inDF.shape[0]}")
	inDF = prepare.createdescriptors(df = inDF, colName = datastore["column_SMILES"]['content'], correlationthreshold = datastore["correlation_threshold"]['content'], STDthreshold = datastore['std_threshold']['content'], IDboolean = IDboolean)
	#print(inDF.head)
	activityValidDF, activityTrainDF, activityTestDF, IDValidDF, IDTrainDF, IDTestDF, validDF, trainDF, testDF, nameValidDF, nameTrainDF, nameTestDF, _ = prepare.partition(df = inDF,validset = datastore['valid_split']['content'], testset = datastore['test_split'] ['content'], IDboolean = IDboolean)
	print("-----------------------------------")
	print("Partitioning Data")
	print("-----------------------------------\n")


	X_Valid = validDF
	Y_Valid = activityValidDF
	X_Train = trainDF
	Y_Train = activityTrainDF
	X_Test = testDF
	Y_Test = activityTestDF


	#print(X_Valid)
	#print(Y_Train)
	# Make a custom metric function
	def my_custom_accuracy(y_true, y_pred):
		return r2_score(y_true, y_pred)
	my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)
	start = timer()
	tpot = TPOTRegressor(generations=25, population_size=25, verbosity=2, cv = 10, n_jobs = -1, use_dask = False, periodic_checkpoint_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/')
	tpot.fit(X_Train, Y_Train)
	print("-----------------------------------")
	print("Saving Predictions...")
	print("-----------------------------------\n")
	Y_Test_Pred = tpot.predict(X_Test)
	Y_Train_Pred = tpot.predict(X_Train)
	Y_Valid_Pred = tpot.predict(X_Valid)

	SMILESTest = []
	YTestList = []
	YTestPredList = []
	SMILESValid = []
	YValidList = []
	YValidPredList = []
	for i in range(0,IDTestDF.shape[0]):
		SMILESTest.append(IDTestDF.loc[:,].values[i])
		YTestList.append(Y_Test.loc[:,].values[i])
		YTestPredList.append(Y_Test_Pred[i])
	for i in range(0,IDTrainDF.shape[0]):
		#NAMESList.append(nameTrainDF.loc[:, ].values[i])
		SMILESTest.append(IDTrainDF.loc[:,].values[i])
		YTestList.append(Y_Train.loc[:,].values[i])
		YTestPredList.append(Y_Train_Pred[i])
	res = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
	SMILESTest = []
	YTestList = []
	YTestPredList = []
	#NAMESList = []
	for i in range(0,IDValidDF.shape[0]):
		#NAMESList.append(nameValidDF.loc[:, ].values[i])
		SMILESTest.append(IDValidDF.loc[:,].values[i])
		YTestList.append(Y_Valid.loc[:,].values[i])
		YTestPredList.append(Y_Valid_Pred[i])
		
	res_valid = pd.DataFrame({'SMILES':SMILESTest,  'Actual':YTestList, 'Prediction':YTestPredList})
	res.to_csv(current_folder + 'predictions/automl_test.csv', sep=',')
	res_valid.to_csv(current_folder + 'predictions/automl_valid.csv', sep=',')
	print(r2_score(Y_Test, Y_Test_Pred))
	end = timer()
	print('---------------------------\n')
	print('TIME')
	time_duration = end - start
	print(f"Time taken: {time_duration}")# Time in seconds, e.g. 5.38091952400282
	tpot.export('tpot_regression.py')


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
	ax.set_xlabel('Leverage')
	ax.set_ylabel('Standardized Residuals')
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

	fig.savefig('visualizations/automLregression.png')
	
	del(res)
	del(res_valid)
	del(X_Train)

	return time_duration, r2_score(Y_Train, Y_Train_Pred), r2_score(Y_Test, Y_Test_Pred), r2_score(Y_Valid, Y_Valid_Pred)