# -*- coding: utf-8 -*-
"""
All functions for computing the wavelet distributions 
"""
from distances.distances_between_graphs import *
import pygsp
import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
from utils.graph_tools import *



def heat_diffusion(G, taus=[1, 10, 25, 50], diff_type="immediate",
                   b=1, type_graph="pygsp"):
    '''
        This method computes the heat diffusion waves for each of the nodes
     INPUT:
    -----------------------
    G: Graph, can be of type networkx or pygsp
    taus: list of 4 scales for the wavelets. The higher the tau, the better the spread
    type: tyoe of the  graph (networkx or pygsp)

    OUTPUT:
    -----------------------
    list_of_heat_signatures: list of 4 pandas df corresponding to the heat signature
    for each diffusion wavelet
    '''
    if type_graph == "pygsp":
        A = G.W
        N = G.N
        if diff_type == "delayed":
            Hk = pygsp.filters.DelayedHeat(G, b, taus, normalize=False)
        elif diff_type == "mexican":
            Nf = 6
            Hk = pygsp.filters.MexicanHat(G, Nf)
        elif diff_type=="wave":
            Hk = pygsp.filters.Wave(G, taus, normalize=False)
        else:
            Hk = pygsp.filters.Heat(G, taus, normalize=False)
            
    elif type_graph == "nx":
        A = nx.adjacency_matrix(G)
        N = G.number_of_nodes()
        Gg = pygsp.graphs.Graph(A,lap_type='normalized')
        if diff_type == "delayed":
            Hk = pygsp.filters.DelayedHeat(Gg,b, taus, normalize=False)
        elif diff_type == "mexican":
            Nf = 6
            Hk = pygsp.filters.MexicanHat(Gg, Nf)
        elif diff_type == "wave":
            Hk = pygsp.filters.Wave(Gg, taus, normalize=True)
        else:
            Hk = pygsp.filters.Heat(Gg, taus, normalize=False)
    else:
        print "graph type not recognized"
        return False
    heat = {i: pd.DataFrame(np.zeros((N,N)), index=range(N))
            for i in range(len(taus))}   
    for v in range(N):
            ### for each node v , create a signal that corresponds to a Dirac of energy
            ### centered around v and whic propagates through the network
            f = np.zeros(N)
            f[v] = 1
            Sf_vec = Hk.analyze(f) ### creates the associated heat wavelets
            Sf = Sf_vec.reshape((Sf_vec.size / len(taus), len(taus)), order='F')
            for  i in range(len(taus)):
                heat[i].iloc[:, v] = Sf[:, i] ### stores in different dataframes the results
    return [heat[i] for i in range(len(taus))]
#return pd.DataFrame.from_dict(heat)

def compare_simple_diffusion(A):
    N = A.shape[0]
    heat_signature = pd.TimeSeries(np.zeros(N), index=range(N))
    A2 = A.dot(A)
    normA2 = np.array([1.0 / np.sum(A2[i, :]) for i in range(N)])
    normA2 = np.diag(A2)
    A2 = normA2.dot(A2)
    A3 = A2.dot(A)
    normA3 = np.array([1.0 / np.sum(A3[i, :]) for i in range(N)])
    normA3 = np.diag(A3)
    A3 = normA3.dot(A3)
    A4 = A3.dot(A)
    normA4 = np.array([1.0 / np.sum(A4[i, :]) for i in range(N)])
    normA4 = np.diag(A4)
    A4 = normA4.dot(A4)
    A_tot = A + A2 + A3 + A4
    for v in range(N):
        heat_signature.loc[v] = np.sum(A[i, :])


def compute_heat_diff(node1, node2, mode, list_heat_df, plot=False, savefig=False):
    test1 = np.sort((list_heat_df[mode]).loc[:,node1])
    test2 = np.sort((list_heat_df[mode]).loc[:,node2])
    n = len(list_heat_df[mode])
    ### delta
    w1 = np.array([(test1[i+1] - test1[i]) for i in range(n-1)])
    w2 = np.array([(test2[i+1] - test2[i]) for i in range(n-1)])
    m1 = np.array([0.5 * (test1[i+1] + test1[i]) for i in range(n-1)])
    m2 = np.array([0.5 * (test2[i+1] + test2[i]) for i in range(n-1)])
    #area1=[(delta2[i]-delta1[i])*m1[i]-delta2[i]*test1[i]+test2[i]*delta1[i] for i in range(n-1)]
    area1=[ 0.5 * np.abs(m2[i] - m1[i]) * (w1[i] + w2[i]) for i in range(n-1)]
    if plot==True:
            plt.plot(test1, test2, c="red", label=str(node1)+" vs " +str(node2))
            plt.plot(test1, test1, c="black", label="45 degree line")
            plt.title("qqplot of node "+ str(node1)+ " vs  node " +str(node2))
            plt.legend(loc="upper left")
            plt.xlabel("node "+str(node1))
            plt.ylabel("node "+str(node2))
            if savefig is True:
                plt.savefig("plots/qq_heat_profile"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    
    return np.sum(area1)

