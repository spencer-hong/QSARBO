'''
Network visualization class
'''

# Imports
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.neighbors as skn

class NetworkVisualization:
    '''
    Class containing methods to perform network visualization.
    '''

    def __init__(self,a_data,**kwargs):
        '''
        Constructor.

        INPUT
            a_data: (pandas Data Frame) Data frame containing the data to visualize. Should be formatted such that the first column is the activity while the remaining columns are the descriptors.

            **kwargs: (dict) Dictionary containing information about the type of model desired and other model-specific arguments.

                'model' :   'knn'   # Specify type of model
                'knn'   :   int     # Specify number of nearest neighbors for knn model
                'plot_type' :   'spring'  # Type of netword plot
                'labels' : {}   # Labels for nodes

        OUTPUT
        '''

        # Convert data to pandas data frame if given in numpy format
        if (type(a_data) == np.ndarray):
            data_copy = pd.DataFrame(a_data)
            self.dataDF = TrainDF_copy.copy()
        # Store data
        else:
            self.dataDF = a_data.copy()

        # Default kwargs
        self.kwargs = {'model':'knn',
                       'knn': 2,
                       'plot_type' : 'spring',
                       'labels' : {}}

        # Set key word arguments
        for key in kwargs:
            self.kwargs[key] = kwargs[key]

        # Check to make sure kwargs are consistent

    def visualize(self):
        '''
        Visualize the data using a network.
        '''

        # Set up model
        # KNN
        if (self.kwargs['model'] == 'knn'):
            graph,values = self.knn(self.dataDF,self.kwargs['knn'])

        # Visualize graph
        self.plot(graph,values,**self.kwargs)

    def plot(self,graph,values,**kwargs):
        '''
        Plot graph.

        INPUT
            graph: (NetworkX graph) Assembled graph.

            values: (list of floats) Values to assign to each data point for color.

            **kwargs: (dict) Plotting key word arguments.
        '''

        # Set plot type
        if (kwargs['plot_type'] == 'spring'):
            pos = nx.spring_layout(graph)
        elif (kwargs['plot_type'] == 'circular'):
            pos = nx.circular_layout(graph)
        elif (kwargs['plot_type'] == 'shell'):
            pos = nx.shell_layout(graph)
        elif (kwargs['plot_type'] == 'spectral'):
            pos = nx.spectral_layout(graph)

        # Set up nodes
        nodes = graph.nodes()

        # Plot
        ec = nx.draw_networkx_edges(graph, pos, alpha=0.5)
        nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=values,
                                    with_labels=False, node_size=50, cmap=plt.cm.jet_r)
        nl = nx.draw_networkx_labels(graph, pos, labels=kwargs['labels'])

        plt.colorbar(nc)
        plt.axis('off')
        plt.show()

    def knn(self,a_dataDF,k,lf=2):
        '''
        Setup knn graph.
        '''

        # Variables
        edgeList = []
        activities = a_dataDF.iloc[:,0].values

        # Calculate k nearest neighbors
        tree = skn.KDTree(a_dataDF.iloc[:,1:], leaf_size=lf)
        dist, ind = tree.query(a_dataDF.iloc[:,1:], k=k+1)

        # Convert to a list for deleting first elements
        dist = dist.tolist()
        ind = ind.tolist()

        # Remove self-distance indices
        for index,cmp in enumerate(ind):
            ind[index] = np.delete(ind[index],0)
            dist[index] = np.delete(dist[index],0)

        # Convert index and distance lists into lists of values
        ind = [list(x) for x in ind]
        dist = [list(x) for x in dist]

        # Create graph
        G = nx.Graph()

        # Add nodes
        for index,val in enumerate(activities):
            G.add_node(index,activity=val)

        # Create edge list
        for index in range(len(ind)):
            for index2 in ind[index]:
                edgeList.append((index,index2))

        # Add edges
        G.add_edges_from(edgeList)

        return G,activities

# Main
if (__name__ == '__main__'):
    # Load data
    inDF = pd.read_csv("outDF.csv")

    # Set up network class
    netVis = NetworkVisualization(inDF)

    # Visualize
    netVis.visualize()
