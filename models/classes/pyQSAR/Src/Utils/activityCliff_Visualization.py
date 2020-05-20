'''
Activity cliff visualization module.
'''

# Imports
import numpy as np
import pandas as pd
from ..Utils import c_networkVisualization as netVis
from ..Validation import modelability as modi

def modi_networks(dataDF,actCliffs):
    '''
    Show network graphs of the original data with activity cliffs labeled.

    INPUT
        dataDF: (pandas Data Frame) Data frame containing compound activities as well as the molecular descriptors.

        actCliffs: (list of ints) List the indices of the activity cliffs.
    '''

    # Set up labels
    labels = {}

    for val in actCliffs:
        labels[val] = 'AC'

    # Variables
    nc_kwargs = {'model':'knn','knn': 2,'plot_type' : 'spring', 'labels':labels}

    # Set up network classes
    netClass_orig = netVis.NetworkVisualization(dataDF,**nc_kwargs)

    # Plot activity network
    netClass_orig.visualize()

# Main
if (__name__ == '__main__'):
    pass
