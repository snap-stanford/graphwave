from characteristic_functions import *
from heat_diffusion import *
import graphwave
import networkx as nx
import numpy as np
import pygsp
from shapes import *
from utils.graph_tools import *
from utils.utils import *


def graphwave(graph, taus, time_pnts=range(0, 100, 2), type_graph="nx",
              verbose=False, **kwargs):
    ''' wrapper function for computing the structural signatures using GraphWave
    INPUT
    --------------------------------------------------------------------------------------
    graph      :   nx or pygsp Graph
    time_pt    :   time points at which to evaluate the characteristic function
    taus       :   list of scales that we are interested in. Alternatively,
                   'automatic' for the automatic version of GraphWave
    type_graph :   type of the graph used (either one of 'nx' or 'pygsp')
    verbose    :   the algorithm prints some of the hidden parameters
                   as it goes along
    OUTPUT
    --------------------------------------------------------------------------------------
    chi        :  embedding of the function in Euclidean space
    heat_print :  returns the actual embeddings of the nodes
    taus       :  returns the list of scales used.
    '''
    if type(taus) == str and taus == 'automatic':
        taus = list(np.arange(0.5, 3.0, 0.2))+list(range(3, 5))
        # Compute the optimal embedding
        pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(graph),
                                         lap_type='normalized')
        pygsp_graph.compute_fourier_basis(recompute=True)
        # safety check to ensure that the graph is indeed connected
        l1 = np.where(pygsp_graph.e > 0.1 / pygsp_graph.N)
        l1 = pygsp_graph.e[l1[0][0]]
        smax = -np.log(0.90) * np.sqrt(pygsp_graph.e[-1] / l1)
        smin = -np.log(0.99) * np.sqrt(pygsp_graph.e[-1] / l1)
        if np.sum(taus > smax) > 0:
            smax = np.where(taus > smax)[0][0]
        else:
            smax = len(taus)
        if np.sum(taus < smin) > 0:
            smin = np.where(taus < smin)[0][-1]
        else:
            smin = 0
        if verbose: print 'smax=', smax, ' and smin=', smin
        taus = taus[smin: smax]

    # Compute the heat wavelets
    heat_print, _ = heat_diffusion(graph, list(taus), diff_type='heat',
                                   type_graph=type_graph)
    chi = featurize_characteristic_function(heat_print, time_pnts)

    return chi, heat_print, taus
