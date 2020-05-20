'''
database

DESCRIPTION
    Contains functions to deal with database mining for generating a set of compounds used in modeling.

TO DO
    - Read from flat file databases.
        - Support for CSV.
    - Read from online databases.
'''

# Imports
import pandas as pd
import copy

# Functions
def loadDatabase(dbName):
    '''
    Load a database.

    INPUT
        dbName: (str) Name of database.

    OUTPUT
        outDB: (pandas DataFrame) DataFrame of database with all data.
    '''

    # Read whole database
    outDB = pd.read_csv(dbName)

    return outDB

def DBSelect(dbDF,selectParams):
    '''
    Select columns of a database. All compounds with incomplete information will be removed from the set.

    INPUT
        dbDF: (pandas DataFrame) Dataframe corresponding to a loaded database.
        selectParams: (List of str) List of strings corresponding to the column titles in the DataFrame.

    OUTPUT
        outDB: (pandas DataFrame) DataFrame of database with requested data.
    '''

    # Select columns
    outDB = copy.deepcopy(dbDF[selectParams])

    # Remove rows with incomplete information
    outDB = outDB.dropna()

    return outDB

def prepareDF(dbDF,smilesFile):
    '''
    Prepare DataFrame for descriptor analysis. Most use cases will be to determine SMILES information.

    INPUT
        dbDF: (pandas DataFrame) Dataframe corresponding to a loaded database.
        smilesFile: (str) Name of file in which SMILES information is contained.

    OUTPUT
        outDB: (pandas DataFrame) DataFrame of database with requested data.

    WARNING
        Not implemented, yet!
    '''

    # Load SMILES database
    smilesDB = loadDatabase(smilesFile)

    pass


# Main
if (__name__ == '__main__'):
    pass
