import sys, os
import json
import pandas as pd 

sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import runner_rf
from models import combi
from models import combi_c
from models import runner_nn
from models import saved_c
from models import automl
from models import saved
from timeit import default_timer as timer
del sys.path[0]

if 'input.json':
	with open('input.json', 'r') as f:
	    datastore = json.load(f)
if datastore['developer_mode?']['content'] == 1:
	## we are going to run 5 runs of combinatorial QSAR, 5 runs of autoML and 5 runs of other future models.
	df = pd.DataFrame(columns = ['type', 'time', 'train', 'test', 'valid'])
	for i in range(0, 5):
		#time, train, test, valid = runner_rf.runner_rf('input.json')
		time, train, test, valid = automl.tpot_r('input.json')
		df.loc[i] = ['automl']  + [time, train, test, valid]
	for i in range(5, 10):
		#time, train, test, valid = runner_rf.runner_rf('input.json')
		time, train, test, valid = runner_rf.runner_rf('input.json')
		df.loc[i] = ['Random Forest']  + [time, train, test, valid]
	for i in range(10, 15):
		#time, train, test, valid = runner_rf.runner_rf('input.json')
		time, train, test, valid = runner_nn.runner_nn('input.json')
		df.loc[i] = ['Neural Network']  + [time, train, test, valid]
	print(df)
elif datastore['test or train?']['content'] == 1:
	print('-----------')
	print('You have chosen to train a model.')
	print('-----------')
	if datastore['which_models?']['content'] == 1:
		print('-----------')
		print('You have chosen to train with random forest.')
		print('-----------')
		if datastore['classification or regression?']['content'] == 'classification':
			print('-----------')
			print('You have chosen to run classification.')
			print('-----------')
			if datastore["autoML?"]['content'] == 1:
				print('automl initiated')
				print(automl.tpot_c('input.json'))
			else:
				print(runner_rf.runner_rf_c('input.json'))
		else:
			print('-----------')
			print('You have chosen to run regression.')
			print('-----------')
			if datastore["autoML?"]['content'] == 1:
				print('automl initiated')
				print(automl.tpot_r('input.json'))
			else:
				print(runner_rf.runner_rf('input.json'))
	elif datastore['which_models?']['content'] == 2:
		print('-----------')
		print('You have chosen to train with neural network.')
		print('-----------')
		#print(runner_nn.runner_nn('input.json'))
		if datastore['classification or regression?']['content'] == 'classification':
			print('-----------')
			print('You have chosen to run classification.')
			print('-----------')
			print(runner_nn.runner_nn_c('input.json'))
		else:
			print('-----------')
			print('You have chosen to run regression.')
			print('-----------')
			print(runner_nn.runner_nn('input.json'))

	else:
		if datastore["autoML?"]['content'] == 1:
			if datastore['classification or regression?']['content'] == 'classification':
				print('automl initiated')
				print(automl.tpot_c('input.json'))
			else:
				print('automl initiated')
				print(automl.tpot_r('input.json'))
		else:
			print('-----------')
			print('You have chosen to train with both random forest and neural network. CombiQSAR will automatically apply.')
			print('-----------')
			if datastore['classification or regression?']['content'] == 'regression':
				start = timer()
				print(runner_rf.runner_rf('input.json'))
				print(runner_nn.runner_nn('input.json'))
				end = timer()
				print('---------------------------\n')
				print('TIME')
				print(end - start)
				print('---------------------------\n')
				current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
				print(combi.combi(current_folder))
			else:
				start = timer()
				print(runner_rf.runner_rf_c('input.json'))
				print(runner_nn.runner_nn_c('input.json'))
				end = timer()
				print('---------------------------\n')
				print('TIME')
				print(end - start)
				print('---------------------------\n')
				current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'+datastore["folder_name"]["content"] +'/'
				print(combi_c.combi_c(current_folder))
else:
	print('-----------')
	print('You have chosen to test with unknown compounds. Make sure that the CSV containing your unknown chemicals is correctly typed in the dataset_name of the input JSON file.')
	print('-----------')
	if datastore['classification or regression?']['content'] == 'classification':
		print(saved_c.saved_c(datastore, datastore['which_models?']['content']))
	else:
		print(saved.saved(datastore, datastore['which_models?']['content']))