'''
Calculate molecular descriptors.
'''

# Imports
import pandas as pd
import rdkit
import mordred
from mordred import descriptors as mdesc
import numpy as np
from rdkit import Chem

# Functions
def removeSTD(inDF,thresh=0.25):
    '''
    Removes descriptors with low standard deviation.

    INPUT
        inDF: (pandas Data Frame) Data frame with molecule SMILES.

        thresh: (float) Threshold for removal.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with descriptors with low standard deviation removed.
    '''

    # Variables
    outDF = inDF.copy()

    # Remove columns
    outDF = outDF.drop(outDF.iloc[:,1:].std()[outDF.iloc[:,1:].std() < thresh].index.values, axis=1)

    return outDF

def removeCorrelated(inDF,thresh=0.25):
    '''
    Removes correlated descriptors.

    INPUT
        inDF: (pandas Data Frame) Data frame with molecule SMILES.

        thresh: (float) Threshold for removal.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with correlated descriptors removed.

    NOTES
        Adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
    '''

    # Variables
    outDF = inDF.copy()

    # Calculate correlation matrix
    corr_matrix = outDF.iloc[:,1:].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find name of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > thresh)]

    # Drop features
    outDF = outDF.drop(labels=to_drop, axis=1)

    '''
    # Plot correlation matrix
    import matplotlib.pyplot as plt
    plt.matshow(corr_matrix)
    plt.show()
    '''

    return outDF

def cleanDescriptors(inDF,descStart=2):
    '''
    Clean up descriptors by removing non-numerical columns.

    INPUT
        inDF: (pandas Data Frame) Data frame with molecule SMILES.

        descStart: (int) Index of the column where descriptors begin.

    OUTPUT
        outDF: (pandas Data Frame) Data frame of original DF with cleaned descriptors.
    '''

    # Variables
    outDF = inDF.copy()
    outDF = outDF.replace([np.inf, -np.inf], np.nan)
    rmColName = []

    # Remove null values
    nulls = np.where(pd.isna(outDF))
    outDF = outDF.drop(outDF.columns[nulls[1]],axis=1)

    # Get column names
    colNames = list(outDF.columns.values)

    # Loop through columns and check for non-numerical values
    for index,col in enumerate(colNames[descStart:]):
        for colVal in inDF[col].values:
            
            # Non-numerical values
            try:
                float(colVal)
            except ValueError:
                rmColName.append(col)
                break
            except TypeError:
                break

            # NaN values
            if (float(colVal) is np.nan):
                rmColName.append(col)
                break

            # Boolean values
            if ((str(colVal).lower() == 'true') or (str(colVal).lower() == 'false')):
                rmColName.append(col)
                break
            
            '''
            # Too large of values
            if (inDF[col].max() > 10e10):
                rmColName.append(col)
                break
            '''

    # Remove columns
    outDF = outDF.drop(labels=rmColName,axis=1)

    return outDF

def calc_mordred(inDF,colName='Structure',coord=2):
    '''
    Calculate Mordred descriptors.

    INPUT
        inDF: (pandas Data Frame) Input data frame with structure information.

        colName: (str) Name of column with structure information.

        coord: (int) 2 for 2D. 3 for 3D.

    OUTPUT
        outDF: (pandas Data Frame) Data Frame with descriptors added as columns.
    '''

    # Copy input data frame
    outDF = inDF.copy()

    # Set up molecules in rdkit format
    strList = outDF[colName].values

    # 2d descriptors
    if (coord == 2):
        # Set up descriptor calculator
        calc = mordred.Calculator(mdesc, ignore_3D=True)

        mols = [rdkit.Chem.MolFromSmiles(smi) for smi in strList]
    # 3d descriptors
    else:
        # Set up descriptor calculator
        calc = mordred.Calculator(mdesc, ignore_3D=False)
        
        # Combine lines
        tmpList = []

        for index,molecule in enumerate(strList):
            tmpList.append(strList[index][0])

        # Save SDF to temp file
        with open('tmpSDF.sdf', 'w') as outFile:
            outFile.write(''.join(tmpList))

        # Load RDKit SDF supplier
        mols = rdkit.Chem.SDMolSupplier('tmpSDF.sdf')
    # Calculate descriptors
    # for mol in mols:
    #     if mol is None: 
    #         print('messed up')
    #         print(mol.getAtoms())
    for i in range(0, len(mols)):
        try:
            Chem.SanitizeMol(mols[i])
        except:
            print('--------------------')
            print(f"This {mols[i]} failed. You must manually fix this SMILE.")
            print(f"The molecule to be fixed is in {i}th molecule.")
            print('--------------------')
    descriptorDF = calc.pandas(mols,quiet=True)

    # Add descriptor columns to outDF
    descriptorDF = descriptorDF.set_index(outDF.index.values)
    outDF = pd.concat([outDF, descriptorDF], axis=1, sort=False)
    print(outDF)
    print('-------------------------')
    print('-------------------------')
    return outDF

# Main
if (__name__ == '__main__'):
    import curate

    # Load test file
    inDF = pd.read_csv('/Users/danielburrill/Research/ChMx/Programming/pyQSAR/Test/MMP_2.csv')

    # Format data frame
    inDF = curate.removeDuplicateStructures(inDF,colName='MOLECULE(SMILES)')
    inDF = curate.removeInvalidSmiles(inDF,colName='MOLECULE(SMILES)')
    inDF = curate.removeSalts(inDF,colName='MOLECULE(SMILES)')

    # Generate 3d structures
    inDF = curate.smi2sdf_par(inDF,colName='MOLECULE(SMILES)')

    # Calculate descriptors
    #inDF = calc_mordred(inDF,colName='MOLECULE(SMILES)')
    inDF = calc_mordred(inDF,colName='SDF',coord=3)

    inDF.to_csv('output.csv')
