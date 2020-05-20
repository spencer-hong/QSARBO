import pandas as pd 
import numpy as np 
import sklearn.metrics as me 
import pickle

def combi(current_folder):

	nn_test = pd.read_csv(current_folder + 'predictions/nn_test.csv')
	rf_test = pd.read_csv(current_folder + 'predictions/rf_test.csv')

	nn_test.sort_values(by=['SMILES'], inplace=True)
	rf_test.sort_values(by=['SMILES'], inplace = True)
	rf_valid_score  = pickle.load( open( current_folder + "pickled/rf_validscore.p", "rb" ) )
	nn_valid_score  = pickle.load( open( current_folder + "pickled/nn_validscore.p", "rb" ) )

	weights_i = []
	weights_j = []
	scores = []
	for i in np.linspace(0, 1, 100):
		for j in np.linspace(0, 1, 100):
			weights_i.append(i)
			weights_j.append(j)
			y_pred_test = nn_test['Prediction'] * j + rf_test['Prediction'] * i
			score = me.r2_score(rf_test['Actual'], y_pred_test)
			scores.append(score)

	best_weight_i = weights_i[scores.index(max(scores))]
	best_weight_j = weights_j[scores.index(max(scores))]

	print(f"The best weight for Random Forest was: {best_weight_i}")
	print(f"The best weight for Neural Network was: {best_weight_j}")


	nn_valid = pd.read_csv(current_folder + 'predictions/nn_valid.csv')
	rf_valid = pd.read_csv(current_folder + 'predictions/rf_valid.csv')

	nn_valid.sort_values(by=['SMILES'], inplace=True)
	rf_valid.sort_values(by=['SMILES'], inplace = True)

	y_pred_valid = nn_valid['Prediction'] * best_weight_j + rf_valid['Prediction']*best_weight_i
	y_pred_test = nn_test['Prediction'] * best_weight_j + rf_test['Prediction']*best_weight_i
	combi_valid_score = me.r2_score(nn_valid['Actual'], y_pred_valid)

	if combi_valid_score <= rf_valid_score and rf_valid_score >= nn_valid_score:
		print("-----------------------")
		print(f"Combi_QSAR did not improve the validation score. The NN valid score was {nn_valid_score}. The RF valid score was {rf_valid_score}. The combiQSAR valid score was {combi_valid_score}. We will use the RF weights.")
		print("-----------------------")
		weightdict = {
		
		'best weight nn':0,
		'best weight rf':1
		}
	elif combi_valid_score <= nn_valid_score and nn_valid_score >= rf_valid_score:
		print("-----------------------")
		print(f"Combi_QSAR did not improve the validation score. The NN valid score was {nn_valid_score}. The RF valid score was {rf_valid_score}. The combiQSAR valid score was {combi_valid_score}. We will use the NN weights.")
		print("-----------------------")
		weightdict = {
		
		'best weight nn':1,
		'best weight rf':0
		}
	else:
		weightdict = {
		
		'best weight nn':best_weight_j,
		'best weight rf':best_weight_i
		}
		print("-----------------------")
		print(f"Combi-QSAR improved the validation score from {nn_valid_score} (NN) and {rf_valid_score} (RF) to {combi_valid_score}.")
		print("-----------------------")


	pickle.dump(weightdict, open(current_folder + "pickled/combi.p", "wb"))

	SMILES = []
	Actual = []
	Pred_rf = []
	Pred_nn = []
	Pred_combi = []
	#NAMESList = []
	smilescolumn = int(nn_test.columns.get_loc('SMILES'))
	actualcolumn = int(nn_test.columns.get_loc('Actual'))
	predictioncolumn = int(nn_test.columns.get_loc('Prediction'))


	for i in range(0,nn_test.shape[0]):
		#NAMESList.append(nameValidDF.loc[:, ].values[i])
		SMILES.append(nn_test.loc[:, 'SMILES'].values[i])
		Actual.append(nn_test.loc[:,'Actual'].values[i])
		Pred_rf.append(rf_test.loc[:,'Prediction'].values[i])
		Pred_nn.append(nn_test.loc[:,'Prediction'].values[i])
		Pred_combi.append(y_pred_test.iloc[:,].values[i])

	for i in range(0,nn_valid.shape[0]):
		#NAMESList.append(nameValidDF.loc[:, ].values[i])
		SMILES.append(nn_valid.loc[:, 'SMILES'].values[i])
		Actual.append(nn_valid.loc[:,'Actual'].values[i])
		Pred_rf.append(rf_valid.loc[:,'Prediction'].values[i])
		Pred_nn.append(nn_valid.loc[:,'Prediction'].values[i])
		Pred_combi.append(y_pred_valid.iloc[:,].values[i])
			
	res_valid = pd.DataFrame({'SMILES':SMILES,  'Actual':Actual, 'Prediction (RF)':Pred_rf, 'Prediction (NN)':Pred_nn, 'Prediction (combo)':Pred_combi})
	res_valid.to_csv(current_folder + 'predictions/combination.csv', sep=',')

	del(nn_valid)
	del(rf_valid)
	del(nn_test)
	del(rf_test)
	del(res_valid)

	return 'done'

