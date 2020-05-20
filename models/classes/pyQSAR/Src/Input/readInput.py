'''
READ INPUT FILE

DESCRIPTION
    Read input file.

CURRENT FORMATS SUPPORTED
    - CSV
'''

# Imports
import pandas as pd
import json
import copy

# Functions
def read_CSV(inFileName):
    '''
    Read input from CSV file.

    INPUT
        inFileName: (str) Path of CSV file.

    OUTPUT
        outDF: (pandas Data Frame) Data frame of input information
    '''

    # Read CSV
    outDF = pd.read_csv(inFileName)

    return outDF

def read_json(inFileName,augDict={}):
    '''
    Read input from JSON file.

    INPUT
        inFileName: (str) Path to JSON file.
        
        augDict: (dict) Dictionary to augment with the JSON information.

    OUTPUT
        outVars: (dict) Dictionary of input information.
    '''

    # Create a copy of augDict
    outVars = copy.deepcopy(augDict)

    # Read JSON file
    with open(inFileName) as inFile:
        inVars = json.load(inFile)

    # Augment dictionary
    for key in inVars:
        outVars[key] = inVars[key]

    return outVars

# Main
if (__name__ == '__main__'):
    pass
