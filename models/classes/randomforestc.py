import pandas as pd
import numpy as np 
import sklearn.ensemble as sken
import sklearn.neighbors as skne
import sklearn.metrics as me 
import sklearn.model_selection as mose
import sklearn.preprocessing as skp
from datetime import datetime

class randomforestc():
	def __init__(self,  X_Train, Y_Train, X_Test, Y_Test, n_estimators = 100, max_features = 0.5, max_depth = 0.3, min_samples_split = 2, cv = 7):
		self.n_estimators = int(n_estimators)
		self.max_features = max_features
		self.max_depth = max_depth
		self.min_samples_split = int(min_samples_split)
		self.__x_train, self.__x_test, self.__y_train, self.__y_test = X_Train, X_Test, Y_Train, Y_Test
		self.__model = self.rf_model()
		self.cv = cv 
	# random forest model
	def rf_model(self):
		model = sken.RandomForestClassifier(min_samples_split = self.min_samples_split,  max_depth = self.max_depth, n_estimators = self.n_estimators, max_features = self.max_features)

		return model
	
	# evaluate random forest model
	def rf_evaluate(self):
		start = datetime.now()
		kfold = mose.KFold(n_splits=self.cv, random_state=40)
		results = mose.cross_val_score(self.__model,self.__x_train, self.__y_train, cv=kfold, n_jobs = -1, scoring = 'r2')
		print(f"Cross Validation Trial Results: {results.mean()}")
		#print(f"Cross Validation Trial Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
		end =  datetime.now()
		time_taken = end - start
		print(f"Time Taken: {time_taken} seconds")
		return  results.mean()

	