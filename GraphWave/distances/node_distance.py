# -*- coding: utf-8 -*-
"""

"""

#### Distances based on characteristic functions between nodes:

import pygsp
import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pyemd import emd
import sys,os


from GraphWave.distances.distances_between_graphs import *
from GraphWave.utils.graph_tools import *
from GraphWave.characteristic_functions import *




def compare_node_chi(node1, node2,heat_print1,heat_print2, taus=[],t=[],plot=True, savefig=False,filefig='/Users/cdonnat/Dropbox/GSP/plots_proteins/graph1vs2.pdf'):
    
    ''' Compares the characteristic functions of two nodes on potentialy different graphs with unidentical number of nodes
    Parameters
    ----------
    heat_print1,heat_print2: SGWT for the two nodes
    t: (optional) values where the curve is evaluated
    Returns
    -------
    theta: time series of the associated angle (array)
    '''
    
    N=0
    if len(taus)>0:
        heat_print1=heat_print1[taus]
        heat_print2=heat_print2[taus]
    chi1=featurize_characteristic_function(heat_print1,t=t,nodes=[node1])
    chi2=featurize_characteristic_function(heat_print2,t=t,nodes=[node2])
    N=chi1.shape[0]
    M=chi2.shape[0]
    chi1_t=(M*1.0)/N*chi1-(M-N)*1.0/N ### corrects for the potentially different sizes of the vector
    if type_comp=="global":
        chi1_g=np.mean(chi1_t,0) ## take the mean of the coefficients
        chi1_g=np.reshape( chi1_g,[-1,2],order='C')
        chi2_g=np.mean(chi2,0)
        chi2_g=np.reshape( chi2_g,[-1,2],order='C')
        d=np.linalg.norm(chi1_g-chi2_g)
        ax,fig=plt.subplots()
        sb.set_style('white')
        if plot==True:
            plt.plot(chi1_g[:,1],chi1_g[:,2], c="coral",label="Graph 1")
            plt.plot(chi2_g[:,1],chi2_g[:,2], c="blue",label="Graph 2")
            plt.title("characteristic function of the two graphs")
            plt.legend(loc="upper right")
            if savefig==True:
                plt.savefig(filefig)
            
    elif type_comp=="local":
        d=np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                d=np.linalg.norm(chi1_t[i,:]-chi2[j,:])
    else:
        print "comparison type is not recognized"
        d=np.nan
    return d




