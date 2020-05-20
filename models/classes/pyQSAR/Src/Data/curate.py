'''
CURATE DATA

DESCRIPTION
    Curate data by removing compounds with certain elements or salts.
'''

# Imports
import pandas as pd
import numpy as np
import pybel
import rdkit
from rdkit import Chem


# Functions
def removeNaNActivity(inDF):
    '''
    Remove compounds with NaN activity.

    INPUT
        inDF: (pandas Data Frame) Data frame with activity.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with NaN activity removed.

    NOTE
        Activity should be in the first column.
    '''

    # Copy data frame
    outDF = inDF.copy()

    # Determine name of activity column
    actName = (outDF.columns.values)[0]

    # Check for float
    '''
    for val in outDF[actName].values:
        float(val)
    '''
    #for val in outDF[actName]:
    #    print(val)
    #print(outDF[actName])

    #print(outDF.shape)
    #outDF.dropna(subset=[actName])
    #print(outDF.shape)

    return outDF

def removeEmptyRows(inDF):
    '''
    Remove empty entries.

    INPUT
        inDF: (pandas Data Frame) Data frame with structures.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with empty entries removed.
    '''

    # Copy data frame
    outDF = inDF.copy()

    # Convert empty entries to NaN
    #outDF = outDF.iloc[:,1].replace('', np.nan)

    # Drop NaNs
    outDF = outDF.dropna()

    return outDF

def filterElements(inDF,keepEle=[1,6,8,7,15,16],colName='Structure'):
    '''
    Filter elements using obabel.

    INPUT
        inDF: (pandas Data Frame) Data frame with molecule SMILES.

        keepEle: (list of ints) Atomic number of elements to keep.

        colName: (string) Name of column where structure is held.

    OUTPUT
        outDF: (pandas Data Frame) Data frame of original DF with filtered structures.
    '''

    # Variables
    mols = []
    rmList = []

    # Copy data frame
    outDF = inDF.copy()

    # Get SMILES data
    smilesData = outDF[colName].values
    smilesData = [smiles.strip() for smiles in smilesData]

    # Read smiles data into molecule format
    for x in smilesData:
        try:
            mols.append(pybel.readstring("smi", x))
        except OSError:
            pass

    # Check molecules for unwanted atoms
    for index,molecule in enumerate(mols):
        # Get list of atoms
        atomList = molecule.atoms

        # Check each atom
        for atom in atomList:
            # Remove compounds containing O+ because RDKit does not handle them
            if ((atom.atomicnum == 8) and (atom.formalcharge == 1)):
                rmList.append(index)
                break

            if (atom.atomicnum not in keepEle):
                rmList.append(index)
                break

    # Remove unwanted molecules
    #print(outDF.iat[rmList[0],1])
    outDF = outDF.drop(outDF.index[rmList])

    return outDF

def removeSalts(inDF,colName='Structure'):
    '''
    Remove salts using obabel.

    INPUT
        inDF: (pandas Data Frame) Data frame with molecule SMILES.

        colName: (string) Name of column where structure is held.

    OUTPUT
        outDF: (pandas Data Frame) Data frame of original DF with salts removed.
    '''

    # Variables
    mols = []

    # Copy data frame
    outDF = inDF.copy()

    # Get SMILES data
    smilesData = outDF[colName].values
    smilesData = [smiles.strip() for smiles in smilesData]

    # Read smiles data into molecule format
    for x in smilesData:
        try:
            mols.append(pybel.readstring("smi", x))
        except OSError:
            pass
    #mols = [pybel.readstring("smi", x) for x in smilesData]

    # Use obabel to remove salts
    for index,molecule in enumerate(mols):
        mols[index].OBMol.StripSalts()

    smilesData = [molecule.write('smi').strip() for molecule in mols]

    # Add information back into data frame
    outDF[colName] = smilesData

    return outDF

def removeDuplicateStructures(inDF,colName='Structure'):
    '''
    Remove duplicate structures. Checks for indentical entries.

    INPUT
        inDF: (pandas Data Frame) Data frame with structures.

        colName: (string) Name of column where structure is held.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with duplicates removed.
    '''

    # Copy data frame
    outDF = inDF.copy()

    # Drop duplicates
    outDF = outDF.drop_duplicates([colName])

    return outDF

def removeInvalidSmiles(inDF,colName='Structure'):
    '''
    Remove rows with invalid SMILES.

    INPUT
        inDF: (pandas Data Frame) Data frame with structures.

        colName: (string) Name of column where structure is held.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with invalid SMILES removed.
    '''

    # Variables
    mol = ''
    rmIdx = []
    outDF = inDF.copy()

    # Check each SMILES and try to convert it. If fails, then wrong format.
    for index,x in enumerate(outDF[colName]):
        #print(str(index) + '\t' + x)
        try:
            mol = pybel.readstring("smi", x)
            mol = Chem.MolFromSmiles(x)
            Chem.SanitizeMol(mol) 
        except:
            print('Invalid SMILES. Deleted.')
            rmIdx.append((outDF.index)[index])
        

    # Remove rows
    outDF = outDF.drop(index=rmIdx)

    return outDF

def smi2sdf_kernel(smiList):
    '''
    Kernel for parallel execution of SMILES -> SDF.
    '''

    # Imports
    import pybel

    # Variables
    outSDF = []

    # Check to see if smiList is a string or list
    if (isinstance(smiList, list)):
        pass
    else:
        smiList = [smiList]

    # Set up molecules
    mols = [pybel.readstring("smi", x) for x in smiList]

    # Generate 3d coordinates
    for index,molecule in enumerate(mols):
        # Add hydrogens
        mols[index].OBMol.AddHydrogens()

        # Calculate 3d coords
        mols[index].make3D()

        # Save SDF
        outSDF.append(mols[index].write(format='sdf'))

    return outSDF

def smi2sdf_par(inDF,colName='Structure'):
    '''
    Convert SMILES to SDF.

    INPUT
        inDF: (pandas Data Frame) Data frame with structures.
        
        colName: (string) Name of column where structure is held.

    OUTPUT
        outDF: (pandas Data Frame) Data frame with SDF column added.
    '''

    # Imports
    import os
    import multiprocessing as mp

    # Variables
    numProcs = os.cpu_count()
    outDF = inDF.copy()

    # Get list of SMILES
    smilesList = outDF[colName].values

    # Perform conversion
    with mp.Pool(numProcs-1) as p:
        sdfList = p.map(smi2sdf_kernel,smilesList)

    # Create new column
    outDF['SDF'] = sdfList

    return outDF

# Main
if (__name__ == '__main__'):
    inDF = pd.read_csv('/Users/danielburrill/Research/ChMx/Programming/pyQSAR/Test/MMP_2.csv')

    print(inDF.shape)
    inDF = removeDuplicateStructures(inDF,colName='MOLECULE(SMILES)')
    print(inDF.shape)
    inDF = removeInvalidSmiles(inDF,colName='MOLECULE(SMILES)')
    print(inDF.shape)
    inDF = removeSalts(inDF,colName='MOLECULE(SMILES)')
    print(inDF.shape)
    inDF = filterElements(inDF,colName='MOLECULE(SMILES)')
    print(inDF.shape)
