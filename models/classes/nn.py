import pandas as pd
import numpy as np 
import sklearn.ensemble as sken
import sklearn.neighbors as skne
import sklearn.metrics as me 
import sklearn.model_selection as mose
import sklearn.preprocessing as skp

from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from datetime import datetime
import warnings
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['SKLEARN_SITE_JOBLIB']='1'

warnings.filterwarnings("ignore", category=FutureWarning)
class nn():
  def __init__(self, X_Train, Y_Train, X_Test, Y_Test,
                 l1_out=512, 
                 l2_out=512, 
                 l1_drop=0.2, 
                 l2_drop=0.2, 
                 l3_out = 512,
                 l3_drop = 0.2,
                 batch_size=100, 
                 epochs=10, cv = 7):
      self.l1_out = l1_out
      self.l2_out = l2_out
      self.l1_drop = l1_drop
      self.l2_drop = l2_drop
      self.l3_out = l3_out
      self.l3_drop = l3_drop
      self.batch_size = batch_size
      self.epochs = epochs
      self.__x_train, self.__x_test, self.__y_train, self.__y_test = X_Train, X_Test, Y_Train, Y_Test
      self.cv = cv
       
      self.__model = KerasRegressor(build_fn = self.nn_model, nb_epoch = epochs, batch_size = batch_size,  verbose = 0)
        
    
    # mnist model
  def nn_model(self):
      model = Sequential()
      model.add(Dense(self.l1_out, input_shape=(self.__x_train.shape[1], ), kernel_initializer = 'uniform'))
      model.add(Activation('relu'))
      model.add(Dropout(self.l1_drop))
      model.add(Dense(self.l2_out, activation='relu',
                    kernel_initializer = 'uniform'))
      model.add(Dropout(self.l2_drop))
      model.add(Activation('relu'))
      model.add(Dense(self.l3_out, activation = 'relu', kernel_initializer = 'uniform'))
      model.add(Dropout(self.l3_drop))
      model.add(Dense(1))
      model.add(Activation('linear'))
      model.compile(loss='mean_squared_error',
                      optimizer=Adam())

      return model
    
    # evaluate nn model
  def nn_evaluate(self):
        #self.mnist_fit()
    start = datetime.now()
    kfold = KFold(n_splits=self.cv, random_state=40)
    results = cross_val_score(self.__model,self.__x_train, self.__y_train, cv=kfold, n_jobs = 1)
    print(f"Cross Validation Trial Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    #print(f"Cross Validation Trial Results: {results.mean()}")
    end = datetime.now()
    time_taken = end - start
    print(f"Time Taken: {time_taken} seconds")
    return  results.mean()
    # function to run nn class
