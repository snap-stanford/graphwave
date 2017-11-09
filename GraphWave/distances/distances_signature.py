# -*- coding: utf-8 -*-
"""

"""

import pygsp
import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pyemd import emd

import sys,os
sys.path.append('../GraphWave/')
from GraphWave.distances.distances_between_graphs import *
from GraphWave.utils.graph_tools import *
from GraphWave.characteristic_functions import *
#### Set of functions for comparing graphs// lots of plotting options



def compute_auc(dist1,dist2,normalize=True,mode_diff="agg",plot=False, savefig=False,filefig="plots/evolution_qq_heat_profile_nd1.png"):
    ''' Computes the area under curve for comparing two diffusion processes
        ### INPUT:
        ### ===============================================================
        ### dist1: first list or matrix with the values of the diffusion
        ### dist2: other diffusion distribution (raw values, not nec. ranked)
        ### normalize: normalize distributions so that it's in [0,1] x[0,1]?
        ### plot, savefig, filefig: plotting params
        ### OUTPUT:
        ### ===============================================================
        ### area under curve (compares the two distributions)
        '''
    Dist2=[u for u in dist2]
    Dist1=[u for u in dist1]
    if len(Dist2)<len(Dist1):
        Dist2+= [0]*(len(dist1)-len(dist2))
    if len(Dist1)<len(Dist2):
        Dist1+=[0]*(len(dist2)-len(dist1))
    
    test1=np.sort(Dist1)
    test2=np.sort(Dist2)
    n=len(test1)
    if normalize==True:
        test1*=1.0/test1[n-1]
        test2*=1.0/test2[n-1]
    

    ### Compute widths and means
    w1=np.array([(test1[i+1]-test1[i]) for i in range(n-1)])
    w2=np.array([(test2[i+1]-test2[i]) for i in range(n-1)])
    if plot==True:
        #fig,ax=plt.subplots(facecolor="white")
            sb.set_style(style='white')
            sb.plt.plot(test1,test2,c="red",label="node 2 vs node1")
            sb.plt.plot(test1,test1,c="black", label="45 degree line")
            sb.plt.title("qqplot of node "+ str(1))
            sb.plt.legend(loc="upper left")
            sb.plt.xlabel("node "+str(1))
            sb.plt.ylabel("node "+str(2))
            if savefig==True:
                sb.plt.savefig(filefig, transparent=True)
    if mode_diff=="max":
        return np.max(np.abs(test2-test1))
    elif mode_diff=="agg":
        area1=np.array([(test2[i+1])*w1[i] for i in range(n-1)])
        area2=np.array([(test2[i])*w1[i] for i in range(n-1)])
        area3=np.array([(test1[i+1])*w2[i] for i in range(n-1)])
        area4=np.array([(test1[i])*w2[i] for i in range(n-1)])
        return 0.5* (np.sum(area1-area2)+np.sum(area3-area4))
    else:
        print "mode not recognized. Returning aggregation"
        area1=np.array([(test2[i+1])*w1[i] for i in range(n-1)])
        area2=np.array([(test2[i])*w1[i] for i in range(n-1)])
        area3=np.array([(test1[i+1])*w2[i] for i in range(n-1)])
        area4=np.array([(test1[i])*w2[i] for i in range(n-1)])
        return 0.5* (np.sum(area1-area2)+np.sum(area3-area4))

def compute_gen_auc(coeff1,coeff2,normalize=True,mode_diff="agg"):
    ### Computes the distance between two heat prints/diffusion matrices based on node-wise AUC comparison of the diffusion processes
    ### INPUT:
    ### ===============================================================
    ### coeff1:  matrix with the values of the diffusion (each column correspond to a diffusion centered at node i)
    ### coeff2:  second diffusion matrix
    ### OUTPUT:
    ### ===============================================================
    ### area under curve (compares the two distributions) -scores for each node, and agreggated
    N=coeff1.shape[1]
    scores=[None]*N
    for i in range(N):
        scores[i]=compute_auc(coeff1[:,i],coeff2[:,i],normalize=normalize,mode_diff=mode_diff,plot=False, savefig=False)
    return scores, np.sum(scores)

def compute_self_auc(coeff1,normalize=True,mode_diff="agg",plot=False, savefig=False,savetitle="heatmap_coeff1.png"):
    ### Computes similarities between nodes for a single network
    N=coeff1.shape[1]
    scores=np.zeros((N,N))
    for i in range(1,N):
        for j in range(i-1):
            scores[i,j]=compute_auc(coeff1[:,i],coeff1[:,j],normalize=normalize,mode_diff=mode_diff,plot=False, savefig=False)
    if plot==True:
        sb.heatmap(scores+scores.T,cmap="hot")
        if savefig==True:
            plt.savefig("plots/"+ savetitle)

    return scores+scores.T, np.sum(scores)



def distance_nodes(dist, distance='L2'):
    ### Computes the euclidean distance between each pair of nodes based on their feature representation
    ### INPUT:
    ### dist: a matrix where each row represents a feature represnetation of a node
    N=dist.shape[0]
    D=np.zeros((N,N))
    for i in range(1,N):
        for j in range(i):
            if distance=='KS':
                D[i,j]=np.max(np.abs(dist[i,:]-dist[j,:]))
            elif distance=='EMD':
                ### Required params:
                ### P,Q - Two histograms of size H
                ### D_H - The HxH matrix of the ground distance between bins of P and Q
                H=np.max([int(np.ceil(np.sqrt(N))),20])
                hist1,bins_arr=np.histogram(dist[i,:].reshape((-1,)),H)
                #### Normalize histogram
                w=[bins_arr[i+1]-bins_arr[i] for i in range(len(bins_arr)-1)]
                hist1=hist1*1.0/np.matrix(w).dot(hist1)
                hist2,_=np.histogram(dist[j,:].reshape((-1,)),bins_arr)
                hist2=hist2*1.0/np.matrix(w).dot(hist2)
                hist1=np.reshape(np.matrix(hist1), [1, H])
                hist2=np.reshape(np.matrix(hist2), [1, H])
                D_H=np.zeros((H,H))
                for i in range(H):
                    for j in range(H):
                        D_H[i,j]=np.abs(bins_arr[i+1]-bins_arr[j+1])
                
                D[i,j]=emd(np.array(hist1.tolist()[0]),np.array(hist2.tolist()[0]),D_H)
            elif distance=='AUC':
                D[i,j]=compute_auc(dist[i,:],dist[j,:],normalize=False,mode_diff="agg")
            else:
                D[i,j]=np.linalg.norm(dist[i,:]-dist[j,:])
    return D+D.T




def distances_between_signals(sig1,sig2, H=50,distance_type="auc",normalize=True, plot=False,**kwargs):
    '''
        This method computes the distance between 2 signals/ wavelet distributions
        INPUT:
        -----------------------
        sig1, sig2:           the distributions that we want to compare
        H:                    binning parameter (only useful if we want to use the emd distance and actually compare histograms)
        distance_type:        type od distance to use (auc, emd, default: correlation)
        
        OUTPUT:
        -----------------------
        d:                    distance
        '''
    
    hist1,bins_arr=np.histogram(sig1,H)
    #### Normalize histogram
    w=[bins_arr[i+1]-bins_arr[i] for i in range(len(bins_arr)-1)]
    hist1=hist1*1.0/np.matrix(w).dot(hist1)
    hist2,_=np.histogram(sig2,bins_arr)
    hist2=hist2*1.0/np.matrix(w).dot(hist2)
    hist1=np.reshape(np.matrix(hist1), [1, H])
    hist2=np.reshape(np.matrix(hist2), [1, H])

    if plot==True:
        plt.bar(bins_arr[1:],hist1.tolist()[0],width=w,color="salmon", label="node 1",alpha=0.5)
        plt.bar(bins_arr[1:],hist2.tolist()[0],width=w,color="lightblue", label="node 2",alpha=0.5)
        plt.title("Normalized Histograms of the heat signatures for the two signals")
        plt.legend(loc="upper right")
    if distance_type=="default":
        d=np.asscalar(hist1.dot(hist2.T))
    elif distance_type=="emd":
        
        ### Required params:  
        ### P,Q - Two histograms of size H
        ### D - The HxH matrix of the ground distance between bins of P and Q.
        D=np.zeros((H,H))
        for i in range(H):
            for j in range(H):
                D[i,j]=np.abs(bins_arr[i+1]-bins_arr[j+1])
        
        d=emd(np.array(hist1.tolist()[0]),np.array(hist2.tolist()[0]),D)
    elif distance_type=="auc":
        if "mode_diff" in kwargs.keys():
            mode_diff=kwargs["mode_diff"]
        else:
            mode_diff="agg"
        d=compute_auc(sig1,sig2,normalize=normalize,mode_diff=mode_diff,plot=False, savefig=False)
    else:
        print "distance type not implemented"
        d=np.nan
    return d
    

def compare_sig_profiles(node1,node2, mode,list_sig_df):
    '''
    This method compares the signature profiles for each of the nodes
     INPUT:
     -----------------------
    node1, node2:           the nodes that we want to compare
    mode:                   which scale for the signature?
    list_sig_df:            list of diffusion patterns over the graph


    OUTPUT:
    -----------------------
    plots
    '''
    Sf=list_sig_df[mode]
    plt.plot(np.sort(Sf.loc[:,node1]),c="red",label=str(node1))
    plt.plot(np.sort(Sf.loc[:,node2]),c="blue",label=str(node2))
    return True
    #np.sort(Sf.loc[:,node1]),np.sort(Sf[:node2])
    
def compute_distance_matrix(list_sig_df,mode,H=50,distance_type="default",normalize=True,plot=False,savefig=False):
    ''''
    This method compares the signature profiles for every pair of nodes (!!!)
    INPUT:
    -----------------------
    list_sig_df: list of diffusion patterns over the graph
    mode: which scale 
    H: binning parameter (only useful if we want to use the emd distance and actually compare histograms)

    OUTPUT:
    -----------------------
    D=distance_matrix
    '''
    
    N=(list_sig_df[mode]).shape[0] ## number of nodes in the network
    D=np.zeros((N,N))
    for i in range(N-1):
        sig1=list_sig_df[mode].loc[:,i]
        for j in range(i+1,N):
            sig2=list_sig_df[mode].loc[:,j]
            D[i,j]=distances_between_signals(sig1,sig2,H,distance_type=distance_type,normalize=normalize, plot=False)
    D=D+D.T
    if plot==True:
        fig, ax = plt.subplots()
        cax=ax.imshow(D,cmap="hot")
        cbar = fig.colorbar(cax, orientation='vertical')
        ax.set_title("Heatmap for the distances between nodes")
        if savefig==True:
            plt.savefig("plots/distance_matrix_heatmap.png")
        plt.show()
    return D
    
 
def compare_sig_profiles_tot(node1,node2, list_sig_df):
    ''' Test function. Compares signatures of two nodes for  the 3 lowest scales
        INPUT:
        -----------------------
        node1, node2: nodes of interest (colum name)
        list_sig_df: list of diffusion patterns over the graph (dictionary of pandas df --keys correspond to rank of scale (0:smallest)
        
        
        OUTPUT:
        -----------------------
        plot
        '''
    plt.plot(np.sort((list_sig_df[0]).loc[:,node1]), c="red",label="tau1, node "+str(node1))
    plt.plot(np.sort((list_sig_df[0]).loc[:,node2]),c="blue",label="tau1, node "+str(node2))
    plt.plot(np.sort((list_sig_df[1]).loc[:,node1]),':',c="red",label="tau2, node "+str(node1))
    plt.plot(np.sort((list_sig_df[1]).loc[:,node2]),':',c="blue",label="tau2, node "+str(node2))
    plt.plot(np.sort((list_sig_df[2]).loc[:,node1]),'--',c="red",label="tau3, node "+str(node1))
    plt.plot(np.sort((list_sig_df[2]).loc[:,node2]),'--',c="blue",label="tau3, node "+str(node2))
    plt.plot(np.sort((list_sig_df[3]).loc[:,node1]),'-.',c="red",label="tau4, node "+str(node1))
    plt.plot(np.sort((list_sig_df[3]).loc[:,node2]),'-.',c="blue",label="tau4, node "+str(node2))
    plt.legend(loc="upper left",fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node1)+" and "+str(node2) )
    return True   
    

def compare_sig_histograms(node1,node2,mode, list_sig_df):
    ''' Plots a bunch of characteristic functions
        INPUT:
        -----------------------
        list_sig_df: list of diffusion patterns over the graph
        
        OUTPUT:
        -----------------------
        D=distance_matrix
    '''
    plt.hist((list_sig_df[mode]).loc[:,node1], bins=20,color="salmon",label="mode "+str(mode)+" for node "+str(node1),alpha=0.5)
    plt.hist((list_sig_df[mode]).loc[:,node2], bins=20,color="lightblue",label="mode "+str(mode)+" for node "+str(node1),alpha=0.5)
    plt.legend(loc="upper right",fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node1)+" and "+str(node2) )
    return True
    
def plot_bunch_characteristic_functions(heat_print,mode, bunch,color_bunch=[],already_colored=False,names=[],savefig=False,fileplot="plots/plot_characteristic.png"):
    ''' Plots a bunch of characteristic functions
        INPUT:
        -----------------------
        heat_print: dictionary of diffusion patterns over the graph
        mode: which scale
        bunch: list of node ids that we are interested in
        color_bunch: (optional) color that we want the nodes to be plotted in
        names: (optional) node names (useful for the legend)
        OUTPUT:
        -----------------------
        chi: characteristic vector associated to each node in the bunch
        plots
        '''
    
    
    
    
    sb.plt.figure()
    colors=["steelblue","forestgreen","darkturquoise","darkorange","y","indigo","magenta","chocolate","red","blue","lavender","coral", "yellow", "purple","black","pink","lightblue","orange", "yellow", "purple", "violet","tomato", "teal",\
            "grey","olive","peru", "maroon","lime", "green","salmon","aqua","sierra", "magenta","yellowgreen","darkgreen","azure","lightseagreen","palevioletred"]
    if len(color_bunch)==0:
        color_b=[colors[i] for i in range(len(bunch))]
    elif not already_colored:
        color_b=[colors[color_bunch[i]] for i in range(len(bunch))]
    else:
         color_b=color_bunch
    if len(names)==0:
        names={b: "node " + str(b) for b in bunch}
    it=0
    chi={}
    for i in bunch:
        #print i
        if isinstance(heat_print[mode], pd.DataFrame):
            f=characteristic_function(heat_print[mode].iloc[:,i],range(-100,100),False)
        elif isinstance(heat_print, list):
            f=characteristic_function(heat_print[i],[10.0/200*k-5 for k in range(100)],False)
        else:
            print "type of input not recognized"
            return np.nan
        plt.plot(f[:,1],f[:,2], c=color_b[it],label=names[i])
        chi[i]=f
        it+=1
    plt.title("Comparison of the characteristic functions of the diffusion patterns for mode "+ str(mode))
    plt.axis("off")
    plt.legend(loc="upper left")
    if savefig==True:
        plt.savefig(fileplot)
    return chi







def sort_heat_print(list_sig_df) :
    sorted_sig_df=[None]*len(list_sig_df)
    for i in range(len(list_sig_df)):
        sorted_sig_df[i]=list_sig_df[i]
        for j in range((list_sig_df[i]).shape[1]):
            (sorted_sig_df[i]).loc[:,j]=np.sort((list_sig_df[i]).loc[:,j])
    return sorted_sig_df

    
    
    
    
        
        
    
    
