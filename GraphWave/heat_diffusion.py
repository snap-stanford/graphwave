# -*- coding: utf-8 -*-
"""
All functions for computing the wavelet distributions 
"""


import pygsp
import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
from utils.graph_tools import *
from distances.distances_between_graphs import *


def heat_diffusion(G,taus=[1, 10, 25, 50],diff_type="immediate",b=1,type_graph="pygsp"):
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

    if type_graph=="pygsp":
        A=G.W
        N=G.N
        if diff_type=="delayed":
            Hk = pygsp.filters.DelayedHeat(G, b,taus, normalize=False)
        elif diff_type=="mexican":
            Nf=6
            Hk = pygsp.filters.MexicanHat(G, Nf)
        elif diff_type=="wave":
            Hk = pygsp.filters.Wave(G, taus, normalize=False)
        else:
            Hk = pygsp.filters.Heat(G, taus, normalize=False)
            
    elif type_graph=="nx":
        A=nx.adjacency_matrix(G)
        N=G.number_of_nodes()
        Gg = pygsp.graphs.Graph(A,lap_type='normalized')
        if diff_type=="delayed":
            Hk = pygsp.filters.DelayedHeat(Gg,b, taus, normalize=False)
        elif diff_type=="mexican":
            Nf=6
            Hk = pygsp.filters.MexicanHat(Gg, Nf)
        elif diff_type=="wave":
            Hk = pygsp.filters.Wave(Gg, taus, normalize=True)
        else:
            Hk = pygsp.filters.Heat(Gg, taus, normalize=False)
    else:
        print "graph type not recognized"
        return False
    heat={i:pd.DataFrame(np.zeros((N,N)), index=range(N))  for i in range(len(taus))}   
    for v in range(N):
            ### for each node v , create a signal that corresponds to a Dirac of energy
            ### centered around v and whic propagates through the network
            f=np.zeros(N)
            f[v]=1
            Sf_vec = Hk.analyze(f) ### creates the associated heat wavelets
            Sf = Sf_vec.reshape((Sf_vec.size/len(taus), len(taus)), order='F')
            for  i in range(len(taus)):
                heat[i].iloc[:,v]=Sf[:,i] ### stores in different dataframes the results
    return [heat[i] for i in range(len(taus))]
#return pd.DataFrame.from_dict(heat)

def compare_simple_diffusion(A):
    N=A.shape[0]
    heat_signature=pd.TimeSeries(np.zeros(N), index=range(N))
    A2=A.dot(A)
    normA2=np.array([1.0/np.sum(A2[i,:]) for i in range(N)])
    normA2=np.diag(A2)
    A2=normA2.dot(A2)
    A3=A2.dot(A)
    normA3=np.array([1.0/np.sum(A3[i,:]) for i in range(N)])
    normA3=np.diag(A3)
    A3=normA3.dot(A3)
    A4=A3.dot(A)
    normA4=np.array([1.0/np.sum(A4[i,:]) for i in range(N)])
    normA4=np.diag(A4)
    A4=normA4.dot(A4)
    A_tot=A+A2+A3+A4
    for v in range(N):
        heat_signature.loc[v]=np.sum(A[i,:])
        
    
    
    
def compare_heat_profiles(G,list_heat_df,type_graph="nx"):
        ####This method computes the heat diffusion waves for each of the nodes
    #### INPUT:
    #### -----------------------
    #### A: adjacency matrix of the graph
    #### list_heat_df: list of diffusion wavelets over the graph


    #### OUTPUT:
    #### -----------------------
    #### list_of_heat_signatures: list of 4 pandas df corresponding to the heat signature
    #### for each diffusion wavelet
    if type_graph=="nx":
        A=nx.adjacency_matrix(G)
    else:
        A=G.W
    heat_signature={}
    A2=A.dot(A)
    A3=A2.dot(A)
    A4=A3.dot(A)
    N=A.shape[0]
    for v in range(A.shape[0]):  
        heat_signature[v]=np.zeros(N)

    return 


def plot_centered_heat_diffusion(node,mode,G,list_heat_df,type_graph="nx",savefig=False):
    if type_graph=="pygsp":
        G_nx=nx.from_scipy_sparse_matrix(G.W)
    else:
        G_nx=G
    pos=nx.spring_layout(G_nx)
    Sf=list_heat_df[mode]
    nodes=nx.draw_networkx_nodes(G_nx,pos,node_color=Sf.loc[:,node],cmap="hot", label=range(100))
    edges=nx.draw_networkx_edges(G_nx,pos,edge_color="black",width=0.5)
    labels=nx.draw_networkx_labels(G_nx,pos)
    plt.sci(nodes)
    plt.colorbar()
    if savefig==True:
        plt.savefig("plots/heat"+str(mode)+"_nd"+str(node)+".png")
    return True

def compare_heat_profiles(node1,node2, mode,list_heat_df,savefig=False):
    Sf=list_heat_df[mode]
    plt.plot(np.sort(Sf.loc[:,node1]),c="red",label=str(node1))
    plt.plot(np.sort(Sf.loc[:,node2]),c="blue",label=str(node2))
    if savefig==True:
        plt.savefig("plots/comp_heat_profile"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True
    #np.sort(Sf.loc[:,node1]),np.sort(Sf[:node2])

def compare_heat_profiles_qqplot(node1,node2, mode,list_heat_df,savefig=False):
    Sf=list_heat_df[mode]
    plt.plot(np.sort(Sf.loc[:,node1]),np.sort(Sf.loc[:,node2]),c="red",label=str(node1)+" vs " +str(node2))
    plt.plot(np.sort(Sf.loc[:,node1]),np.sort(Sf.loc[:,node1]),c="black", label="45 degree line")
    plt.title("qqplot of node "+ str(node1)+ " vs  node " +str(node2))
    plt.legend(loc="upper left")
    plt.xlabel("node "+str(node1))
    plt.ylabel("node "+str(node2))
    if savefig==True:
        plt.savefig("plots/qq_heat_profile"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True
    #np.sort(Sf.loc[:,node1]),np.sort(Sf[:node2])
 
def compare_heat_profiles_tot(node1,node2, list_heat_df,savefig=False):
    plt.plot(np.sort((list_heat_df[0]).loc[:,node1]), c="red",label="tau1, node "+str(node1))
    plt.plot(np.sort((list_heat_df[0]).loc[:,node2]),c="blue",label="tau1, node "+str(node2))
    plt.plot(np.sort((list_heat_df[1]).loc[:,node1]),':',c="red",label="tau2, node "+str(node1))
    plt.plot(np.sort((list_heat_df[1]).loc[:,node2]),':',c="blue",label="tau2, node "+str(node2))
    plt.plot(np.sort((list_heat_df[2]).loc[:,node1]),'--',c="red",label="tau3, node "+str(node1))
    plt.plot(np.sort((list_heat_df[2]).loc[:,node2]),'--',c="blue",label="tau3, node "+str(node2))
    plt.plot(np.sort((list_heat_df[3]).loc[:,node1]),'-.',c="red",label="tau4, node "+str(node1))
    plt.plot(np.sort((list_heat_df[3]).loc[:,node2]),'-.',c="blue",label="tau4, node "+str(node2))
    plt.legend(loc="upper left",fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node1)+" and "+str(node2) )
    if savefig==True:
        plt.savefig("plots/comp_all_heat_profile"+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True   

def compute_heat_diff(node1,node2, mode,list_heat_df,plot=False, savefig=False):
    test1=np.sort((list_heat_df[mode]).loc[:,node1])
    test2=np.sort((list_heat_df[mode]).loc[:,node2])
    n=len(list_heat_df[mode])
    ### delta
    w1=np.array([(test1[i+1]-test1[i]) for i in range(n-1)])
    w2=np.array([(test2[i+1]-test2[i]) for i in range(n-1)])
    m1=np.array([0.5*(test1[i+1]+test1[i]) for i in range(n-1)])
    m2=np.array([0.5*(test2[i+1]+test2[i]) for i in range(n-1)])
    #area1=[(delta2[i]-delta1[i])*m1[i]-delta2[i]*test1[i]+test2[i]*delta1[i] for i in range(n-1)]
    area1=[ 0.5*np.abs(m2[i]-m1[i])*(w1[i]+w2[i]) for i in range(n-1)]
    if plot==True:
            plt.plot(test1,test2,c="red",label=str(node1)+" vs " +str(node2))
            plt.plot(test1,test1,c="black", label="45 degree line")
            plt.title("qqplot of node "+ str(node1)+ " vs  node " +str(node2))
            plt.legend(loc="upper left")
            plt.xlabel("node "+str(node1))
            plt.ylabel("node "+str(node2))
            if savefig==True:
                plt.savefig("plots/qq_heat_profile"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    
    return np.sum(area1)

def compute_evolution_heat_diff(node1,mode, list_heat_df1,list_heat_df2,type_distance="auc",normalize=True,mode_diff="agg",plot=False, savefig=False):
    
    n=len(list_heat_df1[mode])

    #area1=[(delta2[i]-delta1[i])*m1[i]-delta2[i]*test1[i]+test2[i]*delta1[i] for i in range(n-1)]
    ind1=np.argsort((list_heat_df1[mode]).iloc[:,node1])
    ind2=np.argsort((list_heat_df2[mode]).iloc[:,node1])
    test1=(list_heat_df1[mode]).iloc[ind1,node1]
    test2=(list_heat_df2[mode]).iloc[ind2,node1]
    #print area1
    if plot==True:
        
    
    ### delta
        w1=np.array([(test1[i+1]-test1[i]) for i in range(n-1)])
        w2=np.array([(test2[i+1]-test2[i]) for i in range(n-1)])
        m1=np.array([0.5*(test1[i+1]+test1[i]) for i in range(n-1)])
        m2=np.array([0.5*(test2[i+1]+test2[i]) for i in range(n-1)])
        plt.plot(test1,test2,c="red",label=str(node1) + "at time 0")
        plt.plot(test1,test1,c="black", label="45 degree line")
        plt.title("qqplot of node "+ str(node1))
        plt.legend(loc="upper left")
        plt.xlabel("node "+str(node1)+ "at time 0")
        plt.ylabel("new node "+str(node1)+ "at time 1")
        area1=[ 0.5*np.abs(m2[i]-m1[i])*(w1[i]+w2[i]) for i in range(n-1)]
        if savefig==True:
            plt.savefig("plots/evolution_qq_heat_profile"+str(mode)+"_nd"+str(node1)+".png")
    if type_distance=="auc":
        d=compute_auc(test1,test2,normalize=normalize,mode_diff=mode_diff)
        #d=np.sum(area1)
    elif type_distance=="ks":
        d=np.max(np.abs(np.sort(test1)-np.sort(test2)))
    elif type_distance=="ks_p":
        stats=sc.stats.ks_2samp(test1, test2)
        d=stats[0]
        print d
        print stats[1]
    elif type_distance=="corr":
        test11=(list_heat_df1[mode]).iloc[ind1,node1]
        test12=(list_heat_df2[mode]).iloc[ind1,node1]
        m11=np.array([0.5*(test11[i+1]+test11[i]) for i in range(n-1)])
        m12=np.array([0.5*(test12[i+1]+test12[i]) for i in range(n-1)])
        test21=(list_heat_df1[mode]).iloc[ind2,node1]
        test22=(list_heat_df2[mode]).iloc[ind2,node1]
        m21=np.array([0.5*(test21[i+1]+test21[i]) for i in range(n-1)])
        m22=np.array([0.5*(test22[i+1]+test22[i]) for i in range(n-1)])
    #area1=[(delta2[i]-delta1[i])*m1[i]-delta2[i]*test1[i]+test2[i]*delta1[i] for i in range(n-1)]
        area1=[ 0.5*(np.abs(m12[i]-m11[i])*w1[i]+np.abs(m21[i]-m22[i])*w2[i]) for i in range(n-1)]
        d=np.sum(area1)
    else:
        print "distance type not recognized. Switching to auc"
        d=np.sum(area1)
    
    return d
    

def compare_heat_histograms(node1,node2,mode, list_heat_df,savefig=False):
    plt.hist((list_heat_df[mode]).loc[:,node1], bins=20,color="salmon",label="mode "+str(mode)+" for node "+str(node1),alpha=0.5)
    plt.hist((list_heat_df[mode]).loc[:,node2], bins=20,color="lightblue",label="mode "+str(mode)+" for node "+str(node2),alpha=0.5)
    plt.legend(loc="upper right",fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node1)+" and "+str(node2) )
    if savefig==True:
        plt.savefig("plots/comp_heat_hist"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True

    

def plot_heat_distribution(list_heat_df,node,colors=["salmon", "orange", "lightblue", "green", "pink", "red", "purple", "grey"]):
    for mode in range(len(list_heat_df)):
        plt.hist(1.0/np.max((list_heat_df[mode]).loc[:,node])*(list_heat_df[mode]).loc[:,node], bins=20,color=colors[mode],label="mode "+str(mode)+" for node "+str(node),alpha=0.5)
    plt.legend(loc="upper right",fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node) )    
    return True
