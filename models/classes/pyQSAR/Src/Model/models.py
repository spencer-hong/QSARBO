'''
Train QSAR models

DESCRIPTION
    This module holds functions for training QSAR models.
'''

# Imports
import pandas as pd
import numpy as np
import sklearn.preprocessing as skp
import sklearn.decomposition as skd
import sklearn.ensemble as ske
import sklearn.model_selection as skm
import sklearn.neural_network as skn
import sklearn.metrics as skmet
import imblearn.over_sampling as imbl_over
import imblearn.combine as imbl_comb
from ..Validation.appDom import ad_pdf_normal
from ..Model import c_knnra as knnra

# Functions
def plotPrediction(Y_Train,Y_Train_Pred,Y_Test,Y_pred):
    '''
    Plot measured vs. prediction.
    '''

    # Imports
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Plotting parameters

    mpl.rcParams['font.size'] = 16

    # Variables
    if (len(Y_Test) != 0):
        x_min = min([min(Y_Test),min(Y_pred),min(Y_Train),min(Y_Train_Pred)])
        x_max = max([max(Y_Test),max(Y_pred),max(Y_Train),max(Y_Train_Pred)])
    else:
        x_min = min([min(Y_Train),min(Y_Train_Pred)])
        x_max = max([max(Y_Train),max(Y_Train_Pred)])

    x_min -= 0.05*(x_max-x_min)
    x_max += 0.05*(x_max-x_min)
    xVals = np.linspace(x_min,x_max,1000)

    plt.scatter(Y_Train_Pred,Y_Train,label='Training',color="#003fa0")

    if (len(Y_Test) != 0):
        plt.scatter(Y_pred,Y_Test,label='Testing',color="#b42f21",marker='^')

    plt.plot(xVals,xVals,lw=3,color='k')

    plt.xlim([x_min,x_max])
    plt.ylim([x_min,x_max])

    plt.ylabel(r'Measured Log$_{10}$ RD50 (ppm)')
    plt.xlabel(r'Predicted Log$_{10}$ RD50 (ppm)')
    plt.legend(loc=0)
    plt.tight_layout()

    plt.savefig('Shifted_Regression.pdf',dpi=1000,format='pdf')
    plt.show()

def model_test(TestDF,modelFit):
    '''
    Test data against a model fit.

    INPUT
        TestDF: (pandas Dataframe) Dataframe containing testing data.

        modelFit: (model) Class containing the model which has already been fit.

    OUTPUT

    NOTES
        - The modelFit variable must be a class containing a 'predict' function similar to ScikitLearn model classes.
    '''

    # Variables
    TestDF_cpy = TestDF.copy()

    # Prepare data for prediction
    X_Test = (TestDF_cpy.iloc[:,1:]).values

    # Predict
    Y_Pred = modelFit.predict(X_Test)

    return Y_Pred,X_Test

def model_nn_reg(TrainDF,TestDF):
    '''
    Train regression model using neural network.

    INPUT
        TrainDF: (pandas Data Frame) Training data.

        TestDF: (pandas Data Frame) Testing data.

    OUTPUT
        outDF: (pandas Data Frame) Dataframe containing predicted values.

    NOTES
        Input dataframes should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.
    '''

    # Variables
    TrainDF_cpy = TrainDF.copy()
    TestDF_cpy = TestDF.copy()
    testBool = True

    # Only use test sets with data
    if ((TestDF.shape)[0] == 0):
        testBool = False

    # Get numpy arrays for the activity and descriptors
    X_Train = (TrainDF_cpy.iloc[:,1:]).values
    Y_Train = (TrainDF_cpy.iloc[:,0]).values

    # Only initialize for test sets with data
    if (testBool):
        X_Test = (TestDF_cpy.iloc[:,1:]).values
        Y_Test = (TestDF_cpy.iloc[:,0]).values
    else:
        X_Test = []
        Y_Test = []

    # Set number of hidden layers
    numLayers = (X_Train.shape)[1]

    # Initialize neural network
    mlreg = skn.MLPRegressor(hidden_layer_sizes=(numLayers,),
                             alpha=0.0001,
                             batch_size='auto',
                             learning_rate='constant',
                             learning_rate_init=0.01,
                             solver='lbfgs')

    bagReg = ske.BaggingRegressor(base_estimator=mlreg,
                                  n_estimators=100,
                                  n_jobs=7,
                                  random_state=None,
                                  warm_start=False,
                                  bootstrap=True,
                                  oob_score=True)

    # Fitting
    print("Fitting...")
    bagReg.fit(X_Train,Y_Train)
    Y_Train_Pred = bagReg.predict(X_Train)
    score_train = skmet.r2_score(Y_Train,Y_Train_Pred)
    print("Training: " + str(score_train))

    if (testBool):
        Y_Pred = bagReg.predict(X_Test)
        score_test = skmet.r2_score(Y_Test,Y_Pred)
        print("Testing: " + str(score_test))
    else:
        Y_Pred = []

    # Plot training
    oob_score = bagReg.oob_score_
    print("OOB Score: " + str(oob_score))

    # Plot the results
    print('Plotting...')
    plotPrediction(Y_Train,Y_Train_Pred,Y_Test,Y_Pred)

    return Y_Train_Pred,Y_Train,Y_Pred,Y_Test,bagReg

def model_rf_reg(TrainDF,TestDF):
    '''
    Train regression model using random forest.

    INPUT
        TrainDF: (pandas Data Frame) Training data.

        TestDF: (pandas Data Frame) Testing data.

    OUTPUT
        outDF: (pandas Data Frame) Dataframe containing predicted values.

    NOTES
        Input dataframes should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.
    '''

    # Variables
    TrainDF_cpy = TrainDF.copy()
    TestDF_cpy = TestDF.copy()
    testBool = True

    # Only use test sets with data
    if ((TestDF.shape)[0] == 0):
        testBool = False

    # Get numpy arrays for the activity and descriptors
    X_Train = (TrainDF_cpy.iloc[:,1:]).values
    Y_Train = (TrainDF_cpy.iloc[:,0]).values

    # Only initialize for test sets with data
    if (testBool):
        X_Test = (TestDF_cpy.iloc[:,1:]).values
        Y_Test = (TestDF_cpy.iloc[:,0]).values
    else:
        X_Test = []
        Y_Test = []

    # Modeling
    reg_rf = ske.RandomForestRegressor(random_state=42,
                                       n_estimators=1000,
                                       max_features='auto',
                                       min_samples_split=2,
                                       bootstrap=True,
                                       oob_score=True)

    print(reg_rf)
    # Fitting
    print("Fitting...")
    reg_rf.fit(X_Train,Y_Train)

    Y_Train_Pred = reg_rf.predict(X_Train)
    score_train = skmet.r2_score(Y_Train,Y_Train_Pred)
    print("Training: " + str(score_train))

    if (testBool):
        Y_Pred = reg_rf.predict(X_Test)
        score_test = skmet.r2_score(Y_Test,Y_Pred)
        print("Testing: " + str(score_test))
    else:
        Y_Pred = []

    # Plot training
    oob_score = reg_rf.oob_score_
    print("OOB Score: " + str(oob_score))

    # Print importances
    #cols = TrainDF_cpy.columns.values
    #for index,importance in enumerate(rfreg.feature_importances_):
    #    print(cols[index+1] + ' : ' + str(importance))

    # Plot the results
    print('Plotting...')
    plotPrediction(Y_Train,Y_Train_Pred,Y_Test,Y_Pred)

    '''
    feature_import = zip(TrainDF_cpy.columns.values[1:],reg_rf.feature_importances_)
    feature_import = sorted(feature_import,key=lambda x:x[1])
    feature_import = list(reversed(feature_import))

    for val in feature_import:
        print(val)
    '''

    return Y_Train_Pred,Y_Train,Y_Pred,Y_Test,reg_rf

def model_rf_class(TrainDF,TestDF):
    '''
    Train classification model using random forest.

    INPUT
        TrainDF: (pandas Data Frame) Training data.

        TestingDF: (pandas Data Frame) Testing data.

    OUTPUT
        outDF: (pandas Data Frame) Dataframe containing predicted values.

    NOTES
        Input dataframes should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.
    '''

    # Variables
    TrainDF_cpy = TrainDF.copy()
    TestDF_cpy = TestDF.copy()
    testBool = True

    # Only use test sets with data
    if ((TestDF.shape)[0] == 0):
        testBool = False

    # Get numpy arrays for the activity and descriptors
    X_Train = (TrainDF_cpy.iloc[:,1:]).values
    Y_Train = (TrainDF_cpy.iloc[:,0]).values

    # Only initialize for test sets with data
    if (testBool):
        X_Test = (TestDF_cpy.iloc[:,1:]).values
        Y_Test = (TestDF_cpy.iloc[:,0]).values
    else:
        X_Test = []
        Y_Test = []

    # Set up model
    class_RF = ske.RandomForestClassifier(random_state=42,
                                          n_estimators=1000,
                                          max_features='auto',
                                          min_samples_split=2,
                                          oob_score=True,
                                          class_weight='balanced')

    # Fitting
    print("Fitting...")
    class_RF.fit(X_Train,Y_Train)

    Y_Train_Pred = class_RF.predict(X_Train)

    print("Confusion Matrix - Training:")
    print(set(Y_Train))
    print(skmet.confusion_matrix(Y_Train,Y_Train_Pred))

    if (testBool):
        Y_Pred = class_RF.predict(X_Test)
        print("Confusion Matrix - Testing:")
        print(set(Y_Test))
        print(skmet.confusion_matrix(Y_Test,Y_Pred))
    else:
        Y_Pred = []

    # Output training statistics
    print("OOB Score (Q^2): " + str(class_RF.oob_score_))

    return Y_Train_Pred,Y_Train,Y_Pred,Y_Test,class_RF

def model_rf_class_DEBUG(inDF):
    '''
    Train classification model using random forest.

    INPUT
        inDF: (pandas Data Frame) Input dataframe should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.

    OUTPUT
        outDF: (pandas Data Frame) Single column dataframe containing predicted values.
    '''

    # Variables
    modelDF = inDF.copy()

    #print(modelDF.max())

    # Normalize descriptors
    normDesc = skp.normalize(modelDF.iloc[:,1:])

    # Dimensionality reduction
    pca = skd.PCA(n_components=10)
    #X = pca.fit_transform(normDesc)
    X = normDesc
    Y = (modelDF.iloc[:,0]).values

    # Split into testing and training sets
    X_Train, X_Test, Y_Train, Y_Test = skm.train_test_split(X,Y,
                                                            test_size=0.25,
                                                            random_state=42,
                                                            stratify=None)

    # Check applicability domain
    print('Checking applicability domain...')
    TrainDF = pd.DataFrame(np.hstack((np.matrix(Y_Train).T,X_Train)))
    TestDF = pd.DataFrame(np.hstack((np.matrix(Y_Test).T,X_Test)))
    TestDF = ad_pdf_normal(TestDF,TrainDF)
    X_Test = (TestDF.values)[:,1:]
    Y_Test = (TestDF.values)[:,0]

    # Modeling
    rfclass = ske.RandomForestClassifier(random_state=42,
                                         n_estimators=1000,
                                         max_features='auto',
                                         min_samples_split=2,
                                         oob_score=True,
                                         class_weight='balanced')

    # Fitting
    rfclass.fit(X_Train,Y_Train)
    print("Fitting...")

    #Y_Train_pred = skm.cross_val_predict(rfClass,X_Train,Y_Train,cv=100)
    Y_pred = rfclass.predict(X_Test)
    Y_Train_Pred = rfclass.predict(X_Train)

    # Plot training
    score_train = skmet.r2_score(Y_Train,Y_Train_Pred)
    score_test = skmet.r2_score(Y_Test,Y_pred)
    oob_score = rfclass.oob_score_
    print("Training: " + str(score_train))
    print("Testing: " + str(score_test))
    print("OOB Score: " + str(oob_score))

    # Compute confusion matrix
    print("Confusion Matrix - Training:")
    print(set(Y_Train))
    print(skmet.confusion_matrix(Y_Train,Y_Train_Pred))
    print("Confusion Matrix - Testing:")
    print(set(Y_Test))
    print(skmet.confusion_matrix(Y_Test,Y_pred))

    # Print importances
    #cols = modelDF.columns.values
    #for index,importance in enumerate(rfreg.feature_importances_):
        #print(cols[index+1] + ' : ' + str(importance))

    #plotPrediction(Y_Train,Y_Train_Pred,Y_Test,Y_pred)

    return modelDF

def train_nn_reg_DEBUG(inDF):
    '''
    Train regression model using a neural network.

    INPUT
        inDF: (pandas Data Frame) Input dataframe should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.

    OUTPUT
        outDF: (pandas Data Frame) Single column dataframe containing predicted values.
    '''

    # Variables
    modelDF = inDF.copy()
    numLayers = 100

    # Normalize descriptors
    normDesc = skp.normalize(modelDF.iloc[:,1:])

    # Set number of hidden layers
    numLayers = (normDesc.shape)[1]

    # Dimensionality reduction
    pca = skd.PCA(n_components=10)
    #X = pca.fit_transform(normDesc)
    X = normDesc
    Y = (modelDF.iloc[:,0]).values

    # Split into testing and training sets
    X_Train, X_Test, Y_Train, Y_Test = skm.train_test_split(X,Y,test_size=0.05,random_state=None)

    # Initialize neural network
    mlreg = skn.MLPRegressor(hidden_layer_sizes=(numLayers,),
                             alpha=0.0001,
                             batch_size='auto',
                             learning_rate='constant',
                             learning_rate_init=0.01,
                             solver='lbfgs')

    bagReg = ske.BaggingRegressor(base_estimator=mlreg,
                                  n_estimators=100,
                                  n_jobs=7,
                                  random_state=None,
                                  warm_start=True,
                                  bootstrap=True)

    # Train
    #mlreg.fit(X_Train,Y_Train)
    bagReg.fit(X_Train,Y_Train)

    # Test
    Y_pred = bagReg.predict(X_Test)
    Y_Train_Pred = bagReg.predict(X_Train)
    #Y_pred = mlreg.predict(X_Test)
    #Y_Train_Pred = mlreg.predict(X_Train)

    # Plot training
    score_train = skmet.r2_score(Y_Train,Y_Train_Pred)
    score_test = skmet.r2_score(Y_Test,Y_pred)
    print("Training: " + str(score_train))
    print("Testing: " + str(score_test))
    #print("Training: " + str(bagReg.score(X_Train,Y_Train)))
    #print("Testing: " + str(bagReg.score(X_Test,Y_Test)))
    plotPrediction(Y_Train,Y_Train_Pred,Y_Test,Y_pred)

    return modelDF

def consensus_sampling_class(X_Train,Y_Train,X_Test,Y_Test,model,samplingList):
    '''
    Consensus sampling technique. The idea is to use multiple models to give a better prediction for classification.

    INPUT
        X_Train: (numpy array) Training features.

        Y_Train: (numpy array) Training labels.

        X_Test: (numpy array) Testing features.

        Y_Test: (numpy array) Testing labels.

        model:  (scikitlearn model) Model to use for fitting.

        samplingList: (list of imblearn sampling methods) Sampling methods to use.

    OUTPUT
        Y_Pred: (numpy array) Predicted testing labels.

        cat_stat: (list of floats) Category statistics.
    '''

    # Import
    import copy

    # Variables
    weightList = []
    Y_Testing_Pred_List = []
    Y_Train_C = copy.deepcopy(Y_Train)
    Y_Test_C = copy.deepcopy(Y_Test)
    Y_Pred = np.zeros(len(Y_Test))
    numClasses = len(set(Y_Train))
    weightMatrix = np.zeros((len(Y_Test),numClasses))

    # Set lowest class to 0
    lowClass = min(set(Y_Train_C))

    for index in range(len(Y_Train_C)):
        Y_Train_C[index] = Y_Train_C[index]-lowClass

    for index in range(len(Y_Test_C)):
        Y_Test_C[index] = Y_Test_C[index]-lowClass

    print('Consensus Sampling...')

    # Loop over sampling methods
    for smplNum,sampleMethod in enumerate(samplingList):
        print("Sampling Method: " + str(smplNum))

        # Initialize weight array
        weights = np.zeros(numClasses)

        if (sampleMethod == ''):
            X_Train_Over,Y_Train_Over = X_Train,Y_Train_C
        else:
            # Fit and sample data
            X_Train_Over,Y_Train_Over = sampleMethod.fit_sample(X_Train,Y_Train_C)

        # Training
        model.fit(X_Train_Over,Y_Train_Over)

        # Prediction
        #Y_Train_Pred = model.predict(X_Train_Over)
        Y_Test_Pred = model.predict(X_Test)

        # Calculate confusion matrix
        conMat_Test = skmet.confusion_matrix(Y_Test_C,Y_Test_Pred)
        print(conMat_Test)

        # Calculate weights
        for index in range(numClasses):
            # Correct classification
            weights[index] += conMat_Test[index][index]/np.sum(conMat_Test[index])

        # Add results to appropriate lists
        weightList.append(weights)
        Y_Testing_Pred_List.append(Y_Test_Pred)

    # Determine category for each compound from consensus prediction
    print('Determining Categories...')
    for sampleNum in range(len(samplingList)):
        # Loop over testing compounds
        for compNum in range(len(Y_Test_C)):
            classIdx = Y_Testing_Pred_List[sampleNum][compNum]
            weightVal = weightList[sampleNum][classIdx]
            weightMatrix[compNum,classIdx] += weightVal

    print(weightMatrix)

    # Find prediction from consensus
    print('Find Prediction...')
    for compNum in range(len(Y_Test_C)):
        Y_Pred[compNum] = np.argmax(weightMatrix[compNum,:])

    # Show confusion matrix
    confMat_Pred = skmet.confusion_matrix(Y_Test_C,Y_Pred)
    print('Confusion Matrix - Testing')
    print(set(Y_Test_C))
    print(confMat_Pred)

    # Caclulate statistics
    TPR = []
    accuracy = []
    totalCmpds = 0
    correctCmpds = 0

    for classNum in range(numClasses):
        correctCmpds += confMat_Pred[classNum,classNum]
        totalCmpds += np.sum(confMat_Pred[classNum])
        TPR.append(confMat_Pred[classNum,classNum]/np.sum(confMat_Pred[classNum]))

    print("Accuracy: " + str(1.0*correctCmpds/totalCmpds))
    print("TPR:")
    print(TPR)

    return Y_Pred

def ROC(X_Train, X_Test, Y_Train, Y_Test):
    '''
    Generate ROC diagram. This method relies on the automatic generation of weights to fill the parameter space.
    '''

    # Import
    import matplotlib.pyplot as plt

    # Variables
    coordList = []
    metric = lambda x: x
    weights_0 = metric(np.linspace(0.0,1e-5,30))
    weights_1 = 1-weights_0

    dx = 0.00000001/10.0
    #weights_0 = np.linspace(0,0.00000001,10)
    #weights_0 = np.concatenate([weights_0, np.linspace(0.00000001+dx,0.0000001,10)])
    #weights_0 = np.concatenate([weights_0, np.linspace(0.0000001+dx,0.000001,10)])
    #weights_1 = 1-weights_0
    weights = list(zip(weights_0,weights_1))

    for index in range(len(weights)):
        # Determine weight class
        w = {0:weights[index][0],1:weights[index][1]}
        print(w)

        # Model
        rfClass = ske.RandomForestClassifier(random_state=None,
                                             n_estimators=1000,
                                             max_features='auto',
                                             min_samples_split=2,
                                             oob_score=True,
                                             class_weight=w,
                                             n_jobs=7)

        # Fit
        rfClass.fit(X_Train,Y_Train)

        # Prediction
        Y_Pred = rfClass.predict(X_Test)

        # Calculate confusion matrix
        conMat_Test = skmet.confusion_matrix(Y_Test,Y_Pred)

        # Calculate statistics
        TPR = conMat_Test[1][1]/np.sum(conMat_Test[1])
        FPR = conMat_Test[0][1]/np.sum(conMat_Test[0])
        precision = conMat_Test[1][1]/(conMat_Test[0][1]+conMat_Test[1][1])
        coordList.append((precision,TPR))
        print((precision,TPR))

    # Plot
    xLine = np.linspace(0,1,1000)
    coordList = np.asarray(coordList)
    x,y = coordList.transpose()
    #print(coordList)
    #print('')
    #print(x)
    #print('')
    #print(y)
    #plt.plot(xLine,xLine,color='k')
    plt.scatter(x,y)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

def ModelTesting(inDF):
    '''
    Testing method.

    !!!DEVELOPMENT ONLY!!!
    '''

    print('MODEL TESTING...')

    # Variables
    modelDF = inDF.copy()

    #print(modelDF.max())

    # Normalize descriptors
    normDesc = skp.normalize(modelDF.iloc[:,1:])

    # Dimensionality reduction
    X = normDesc
    Y = (modelDF.iloc[:,0]).values

    # Split into testing and training sets
    X_Train, X_Test, Y_Train, Y_Test = skm.train_test_split(X,Y,
                                                            test_size=0.25,
                                                            random_state=None,
                                                            stratify=None)

    # ROC
    #ROC(X_Train, X_Test, Y_Train, Y_Test)


    # Modeling
    rfclass = ske.RandomForestClassifier(random_state=None,
                                         n_estimators=1000,
                                         max_features='auto',
                                         min_samples_split=2,
                                         oob_score=True,
                                         class_weight=None,
                                         n_jobs=7)

    # Set up sampling list
    samplingList = []

    # SMOTE oversampling
    smote = imbl_over.SMOTE(random_state=None,
                            kind='borderline2',
                            k_neighbors=5,
                            m_neighbors=10,
                            n_jobs=7)

    samplingList.append(smote)

    smote = imbl_over.SMOTE(random_state=None,
                            kind='borderline1',
                            k_neighbors=5,
                            m_neighbors=10,
                            n_jobs=7)

    samplingList.append(smote)

    smote = imbl_over.SMOTE(random_state=None,
                            kind='regular',
                            k_neighbors=5,
                            m_neighbors=10,
                            n_jobs=7)

    samplingList.append(smote)

    # SMOTEENN
    smote = imbl_over.SMOTE(random_state=None,
                            kind='borderline2',
                            k_neighbors=5,
                            m_neighbors=10,
                            n_jobs=7)
    smoteenn = imbl_comb.SMOTEENN(smote=smote)

    samplingList.append(smoteenn)

    # SMOTE Tomek
    smote = imbl_over.SMOTE(random_state=None,
                            kind='borderline2',
                            k_neighbors=5,
                            m_neighbors=10,
                            n_jobs=7)
    smotetomek = imbl_comb.SMOTETomek(smote=smote)

    samplingList.append(smotetomek)

    # Consensus sampling
    consensus_sampling_class(X_Train,Y_Train,X_Test,Y_Test,rfclass,samplingList)

def model_knnra_reg(TrainDF,TestDF,knn=3):
    '''
    Determine activities using k-nearest neighbors read across.

    INPUT
        TrainDF: (pandas Data Frame) Training data.

        TestDF: (pandas Data Frame) Testing data.

        knn: (int) Number of nearest neighbors to use.

    OUTPUT
        Y_Test: (numpy array) Numpy array containing measured values.

        Y_Test_Pred: (numpy array) Numpy array containing predicted values.

    NOTES
        Input dataframes should be structured such that the activity is located in the first column and descriptors/features in all remaining columns.

    REFERENCES
        Willett, Peter, John M. Barnard, and Geoffrey M. Downs. "Chemical similarity searching." Journal of chemical information and computer sciences 38.6 (1998): 983-996.
    '''

    # Variables
    TrainDF_cpy = TrainDF.copy()
    TestDF_cpy = TestDF.copy()
    testBool = True

    # Only use test sets with data
    if ((TestDF.shape)[0] == 0):
        testBool = False

    # Get numpy arrays for the activity and descriptors
    X_Train = (TrainDF_cpy.iloc[:,1:]).values
    Y_Train = (TrainDF_cpy.iloc[:,0]).values

    # Only initialize for test sets with data
    if (testBool):
        X_Test = (TestDF_cpy.iloc[:,1:]).values
        Y_Test = (TestDF_cpy.iloc[:,0]).values
    else:
        X_Test = []
        Y_Test = []

    # Set up model
    class_knnra = knnra.knnRARegressor(knn=knn)

    # Fitting
    print("Fitting...")
    class_knnra.fit(TrainDF_cpy)

    if (testBool):
        Y_Pred = class_RF.predict(TestDF_cpy)
        print("Confusion Matrix - Testing:")
        print(set(Y_Test))
        print(skmet.confusion_matrix(Y_Test,Y_Pred))
    else:
        Y_Pred = []

    return Y_Train,Y_Pred,Y_Test,class_knnra

# Main
if (__name__ == '__main__'):
    pass
