# -*- coding: utf-8 -*-
"""
Distances based on characteristic functions between nodes
"""
from graphwave.distances.distances_between_graphs import *
from graphwave.utils.graph_tools import *
from graphwave.characteristic_functions import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygsp
import seaborn as sb

def compare_node_chi(node1, node2, chi, taus=[],t=[],plot=True, savefig=False,filefig='/Users/cdonnat/Dropbox/GSP/plots_proteins/graph1vs2.pdf'):
    
    ''' Compares the characteristic functions of two nodes on potentialy different graphs with unidentical number of nodes
    Parameters
    ----------
    heat_print1,heat_print2: SGWT for the two nodes
    t: (optional) values where the curve is evaluated
    Returns
    -------
    theta: time series of the associated angle (array)
    '''
    
    n_nodes, _ = chi.shape
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




