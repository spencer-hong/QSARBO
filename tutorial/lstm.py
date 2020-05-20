from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from keras.layers import Dense
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import ExcelFile
from pandas import ExcelWriter
from PIL import Image
from scipy import ndimage
from scipy.stats import randint as sp_randint
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn import metrics
from sklearn import pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.python.framework import ops

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import adam

import sys

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
#from __future__ import print_function
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
 
from matplotlib import pyplot as plt

import keras
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

print('--------------------')
print('    Reading Data     ')
print('--------------------')
data = pd.read_csv(str(sys.argv[1]), error_bad_lines=False)
X_train_smiles = np.array(list(data[str(sys.argv[3])][data["SPLIT"]==1]))
X_test_smiles = np.array(list(data[str(sys.argv[3])][data["SPLIT"]==0]))

trainsize = X_train_smiles.shape[0]
testsize = X_test_smiles.shape[0]


print('--------------------')
print('    Dataset Details     ')
print(f"Training size: {trainsize}")
print(f"Testing size: {testsize}")
print('--------------------')

assay = str(sys.argv[2]) 

Y_train = data[assay][data["SPLIT"]==1].values.reshape(-1,1)
Y_test = data[assay][data["SPLIT"]==0].values.reshape(-1,1)

charset = set("".join(list(data.SMILES))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in data.SMILES]) + 5

print('--------------------')
print('    Character to Integer List     ')
print(char_to_int)
print('--------------------')

def vectorize(smiles):
	one_hot =  np.zeros((smiles.shape[0], embed , len(charset)),dtype=np.int8)
	for i,smile in enumerate(smiles):
		#encode the startchar
		one_hot[i,0,char_to_int["!"]] = 1
		#encode the rest of the chars
		for j,c in enumerate(smile):
			one_hot[i,j+1,char_to_int[c]] = 1
		#Encode endchar
		one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
	#Return two, one for input and the other for output
	return one_hot[:,0:-1,:], one_hot[:,1:,:]


X_train, _ = vectorize(X_train_smiles)
X_test, _ = vectorize(X_test_smiles)

vocab_size=len(charset)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=embed-1))
model.add(keras.layers.Conv1D(192,10,activation='relu'))
model.add(BatchNormalization())
model.add(keras.layers.Conv1D(192,5,activation='relu'))
model.add(keras.layers.Conv1D(192,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='linear'))

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
optimizer = adam(lr=0.00025)
lr_metric = get_lr_metric(optimizer)
model.compile(loss="mse", optimizer=optimizer, metrics=[coeff_determination, lr_metric])

from keras.callbacks import ReduceLROnPlateau
callbacks_list = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-15, verbose=1, mode='auto',cooldown=0),
    ModelCheckpoint(filepath="weights.best.hdf5", monitor='val_loss', save_best_only=True, verbose=1, mode='auto')
    
]



history =model.fit(x=np.argmax(X_train, axis=2), y=Y_train,
                              batch_size=128,
                              epochs=150,
                              validation_data=(np.argmax(X_test, axis=2),Y_test),
                              callbacks=callbacks_list)


Y_pred_train = model.predict(np.argmax(X_train, axis = 2))
Y_pred_test = model.predict(np.argmax(X_test, axis = 2))

trainlist = Y_pred_train.flatten()
testlist = Y_pred_test.flatten()

trainlistactivity = Y_train.flatten()
testlistactivity = Y_test.flatten()

np.append(trainlist, testlist)

np.append(X_train_smiles, X_test_smiles)

np.append(trainlistactivity, testlistactivity)

predictlist = trainlist

smileslist = X_train_smiles

activitylist = trainlistactivity

res= pd.DataFrame({'SMILES':smileslist,  'Actual':activitylist, 'Prediction':predictlist})
res.to_csv('lstm_results.csv', sep=',')

hist = history.history

plt.figure(figsize=(10, 8))

for label in ['val_coeff_determination','coeff_determination']:
    plt.subplot(221)
    plt.plot(hist[label], label = label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("coeff_determination")
    
for label in ['val_loss','loss']:
    plt.subplot(222)
    plt.plot(hist[label], label = label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("loss")



plt.subplot(223)
plt.plot( hist['lr'],hist['val_coeff_determination']  )
plt.legend()
plt.xlabel("lr")
plt.ylabel("val_coeff_determination")


plt.subplot(224)
plt.plot( hist['lr'],hist['val_loss']  )
plt.legend()
plt.xlabel("lr")
plt.ylabel("val_loss")

    
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.savefig('results.png', bbox_inches = 'tight')