import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import matplotlib.pyplot as plt
import sys
import seaborn as sb


from shapes.shapes import *
from heat_diffusion import *
from utils.graph_tools  import *
from utils.utils import *
from characteristic_functions import *
import pygsp
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from utils.graph_tools import *
from heat_diffusion import *
from characteristic_functions import *

def graphwave(G, taus, t=range(0,100,2), type_graph="nx",verbose=False,**kwargs):
    ''' wrapper function for computing the structural signatures using GraphWave
    INPUT
    --------------------------------------------------------------------------------------
    G          :  nx or pygsp Graph 
    taus       :  list of scales that we are interested in. Alternatively, 'automatic'
	               for the automatic version of GraphWave
	type_graph :   type of the graph used (either one of 'nx' or 'pygsp')
    verbose    :   the algorithm prints some of the hidden parameters as it goes along
    OUTPUT
    --------------------------------------------------------------------------------------
    chi        :  embedding of the function in Euclidean space
    heat_print :  returns the actual embeddings of the ndoes
    taus       :  returns the list of scales used.
    '''
    if type(taus)==str:
        taus=[0.5,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.0,2.1,2.3,2.5,2.7]+range(3,5)
        #### Compute the optimal embedding
        Gg=pygsp.graphs.Graph(nx.adjacency_matrix(G),lap_type='normalized')
        Gg.compute_fourier_basis(recompute=True)
        l1=np.where(Gg.e>0.1/Gg.N) ### safety check to ensure that the graph is indeed connected
        l1=Gg.e[l1[0][0]]
        smax=-np.log(0.90)*np.sqrt(Gg.e[-1]/l1)
        smin=-np.log(0.99)*np.sqrt(Gg.e[-1]/l1)
        if np.sum(taus>smax)>0:
            smax=np.where(taus>smax)[0][0]
        else:
            smax=len(taus)
        if np.sum(taus<smin)>0:
            smin=np.where(taus<smin)[0][-1]
        else:
            smin=0
        if verbose: print "smax=",smax, " and smin=", smin
        taus=taus[smin:smax]
    ### Compute the heat wavelets
    heat_print=heat_diffusion(G,taus,diff_type="immediate",type_graph=type_graph)
    chi=featurize_characteristic_function(heat_print,t)
    return chi,heat_print, taus


