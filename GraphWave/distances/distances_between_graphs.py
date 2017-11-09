# -*- coding: utf-8 -*-
"""
Functions for computing distances between graphs
"""

import pygsp
import numpy as np
import networkx as nx 
import pandas as pd
import types
import matplotlib.pyplot as plt
import seaborn as sb
import sys,os

from GraphWave import shapes
from GraphWave.shapes import *
from GraphWave.heat_diffusion import *
from GraphWave.utils.graph_tools import *
from GraphWave.distances.distances_signature import *
from pyemd import emd
import scipy as sc
from GraphWave.characteristic_functions import *
### Define distances between graphs using the wavelets/diffusion profiles:



def distances(G1,G2,type_graphs,type_comp="auc",taus=[1, 10, 25, 50],normalize=True,mode_diff="agg",plot=False,savefig=False,filefig="plots/change_graph.png"):
    '''
    Compare two graphs based on their diffusion properties: assumes that the nodes are identified
     INPUT:
     ======================================================
     G1,G2: two graphs of type either nx or pygsp
     type_graph: array specifying each graph's type  ("nx" or "pygsp")
     type_comp: the distances between distributions that should be used (default: auc)
     taus: the scales used for heat diffusion propoagation
     plot, savefig,filefig: additional parameters (for plotting and saving plot results)
     OUTPUT:
     ======================================================
     D_12: distances between diffusion distribution at different scales
     heat_print1, heat_print2: the heat prints for the two graphs (array of diffusion distributions)
    '''
    if type_graphs[0]=="nx":
        n=G1.number_of_nodes()
    elif type_graphs[0]=="nx":
        n=G1.N
    else:
        print "type graph not recognized. Abort mission"
        return 0
    M=len(taus)
    heat_print1=heat_diffusion(G1,taus=taus,type_graph=type_graphs[0])
    heat_print2=heat_diffusion(G2,taus=taus,type_graph=type_graphs[1])
    D_12= np.zeros((M,n))
    for m in range(M):
        for i in range(n):
            if type_comp=="corr":
                D_12[m,i]=1-1.0/(np.linalg.norm((heat_print1[m]).iloc[:,i])*np.linalg.norm((heat_print2[m]).iloc[:,i]))*((heat_print1[m]).iloc[:,i]).dot((heat_print2[m]).iloc[:,i])
            elif type_comp=="auc":
                D_12[m,i]=abs(compute_evolution_heat_diff(i,m, heat_print1,heat_print2,mode_diff=mode_diff))
            elif type_comp=="emd":
         ### Required params:  
         ### P,Q - Two histograms of size H
         ### D - The HxH matrix of the ground distance between bins of P and Q
                H=30
                hist1,bins_arr=np.histogram(heat_print1[m],H)
                #### Normalize histogram
                w=[bins_arr[i+1]-bins_arr[i] for i in range(len(bins_arr)-1)]
                hist1=hist1*1.0/np.matrix(w).dot(hist1)
                hist2,_=np.histogram(heat_print2[m],bins_arr)
                hist2=hist2*1.0/np.matrix(w).dot(hist2)
                hist1=np.reshape(np.matrix(hist1), [1, H])
                hist2=np.reshape(np.matrix(hist2), [1, H])
                D=np.zeros((H,H))
                for i in range(H):
                    for j in range(H):
                        D[i,j]=np.abs(bins_arr[i+1]-bins_arr[j+1])
        
                D_12[m,i]=emd(np.array(hist1.tolist()[0]),np.array(hist2.tolist()[0]),D)
            else:
                 print "comparison type not implemented"
                 return np.nan
    if plot==True:
        plt.figure()
        sb.heatmap(D_12,cmap="hot")
        if savefig==True:
            plt.savefig(filefig)
    agg_score=np.sum(D_12)
    return D_12,agg_score,heat_print1,heat_print2




    
def distances_heatprints(heat_print1,heat_print2,type_comp="auc",mode_diff="agg",plot=False,savefig=False,filefig="plots/change_graph.png"):
    ''' Compare two heat signatures 
     INPUT:
     ======================================================
     heat_print1,heat_print2: two heat_prints 
     type_comp: the distances between distributions that should be used (default: auc)
     taus: the scales used for heat diffusion propoagation
     plot, savefig,filefig: additional parameters (for plotting and saving plot results)
     OUTPUT:
     ======================================================
     D_12: distances between diffusion distribution at different scales
     heat_print1, heat_print2: the heat prints for the two graphs (array of diffusion distributions)
     '''
     
    taus=[1, 10, 25, 50]
    n=(heat_print1[0]).shape[1]
    M=len(heat_print1)
    D_12= np.zeros((M,n))
    for m in range(M):
        for i in range(n):
            if type_comp=="corr":
                D_12[m,i]=1-1.0/(np.linalg.norm((heat_print1[m]).iloc[:,i])*np.linalg.norm((heat_print2[m]).iloc[:,i]))*((heat_print1[m]).iloc[:,i]).dot((heat_print2[m]).iloc[:,i])
            elif type_comp=="auc":
                D_12[m,i]=abs(compute_evolution_heat_diff(i,m, heat_print1,heat_print2,mode_diff=mode_diff))
            elif type_comp=="emd":
         ### Required params:  
         ### P,Q - Two histograms of size H
         ### D - The HxH matrix of the ground distance between bins of P and Q
                H=30
                hist1,bins_arr=np.histogram(heat_print1[m],H)
                #### Normalize histogram
                w=[bins_arr[i+1]-bins_arr[i] for i in range(len(bins_arr)-1)]
                hist1=hist1*1.0/np.matrix(w).dot(hist1)
                hist2,_=np.histogram(heat_print2[m],bins_arr)
                hist2=hist2*1.0/np.matrix(w).dot(hist2)
                hist1=np.reshape(np.matrix(hist1), [1, H])
                hist2=np.reshape(np.matrix(hist2), [1, H])
                D=np.zeros((H,H))
                for i in range(H):
                    for j in range(H):
                        D[i,j]=np.abs(bins_arr[i+1]-bins_arr[j+1])
        
                D_12[m,i]=emd(np.array(hist1.tolist()[0]),np.array(hist2.tolist()[0]),D)
            else:
                 print "comparison type not implemented"
                 return np.nan
    if plot==True:
        plt.figure()
        sb.heatmap(D_12,cmap="hot")
        if savefig==True:
            plt.savefig(filefig)
    agg_score=np.sum(D_12)
    return D_12,agg_score
    
    
def distances_heatprints_mode(heat_print1,heat_print2,m=2,type_comp="auc",mode_diff="agg",plot=False,savefig=False,filefig="plots/change_graph.png"):
    ### same as distances_heatprints, but only for the specified mode m
    n=(heat_print1[0]).shape[1]
    M=len(heat_print1)
    D_12= np.zeros(n)
    for i in range(n):
        if type_comp=="corr":
            D_12[i]=1-1.0/(np.linalg.norm((heat_print1[m]).iloc[:,i])*np.linalg.norm((heat_print2[m]).iloc[:,i]))*((heat_print1[m]).iloc[:,i]).dot((heat_print2[m]).iloc[:,i])
        elif type_comp=="auc":
            D_12[i]=abs(compute_evolution_heat_diff(i,m, heat_print1,heat_print2,mode_diff=mode_diff))
        else:
             print "comparison type not implemented"
             return 0
    if plot==True:
        plt.figure()
        sb.heatmap(D_12,cmap="hot")
        if savefig==True:
            plt.savefig(filefig)
    agg_score=np.sum(D_12)
    return D_12,agg_score
            
def distances_between_nodes(heat_print,mode,node1,node2,type_comp="auc",mode_diff="agg",normalize="True",plot=False,savefig=False,filefig="plots/nodes_dist.png"):
    ### Computes the distance between two nodes for the same graph/ based on their heat profiles
    if type_comp=="auc":
        d=compute_auc(heat_print[mode].iloc[:,node1],heat_print[mode].iloc[:,node2],normalize=normalize, mode_diff=mode_diff,plot=plot, savefig=savefig,filefig=filefig)
    elif type_comp=="emd":
                 ### Required params:  
         ### P,Q - Two histograms of size H
         ### D - The HxH matrix of the ground distance between bins of P and Q
        H=30
        hist1,bins_arr=np.histogram(heat_print[mode].iloc[:,node1],H)
                #### Normalize histogram
        w=[bins_arr[i+1]-bins_arr[i] for i in range(len(bins_arr)-1)]
        hist1=hist1*1.0/np.matrix(w).dot(hist1)
        hist2,_=np.histogram(heat_print[mode].iloc[:,node2],bins_arr)
        hist2=hist2*1.0/np.matrix(w).dot(hist2)
        hist1=np.reshape(np.matrix(hist1), [1, H])
        hist2=np.reshape(np.matrix(hist2), [1, H])
        D=np.zeros((H,H))
        for i in range(H):
            for j in range(H):
                D[i,j]=np.abs(bins_arr[i+1]-bins_arr[j+1])
        d=emd(np.array(hist1.tolist()[0]),np.array(hist2.tolist()[0]),D)
    elif type_comp=="corr":
        d=1-1.0/(np.linalg.norm((heat_print[mode]).iloc[:,node1])*np.linalg.norm((heat_print[mode]).iloc[:,node2]))*((heat_print[mode]).iloc[:,node1]).dot((heat_print[mode]).iloc[:,node2])
    elif type_comp=="corr_sorted":
        d=1-1.0/(np.linalg.norm((heat_print[mode]).iloc[:,node1])*np.linalg.norm( heat_print[mode].iloc[:,node2]))*(np.sorted(heat_print[mode].iloc[:,node1])).dot(np.sorted(heat_print[mode].iloc[:,node2]))
    elif type_comp=="ks":
        test1=heat_print[mode].iloc[:,node1]
        test2=heat_print[mode].iloc[:,node2]
        d=np.max(np.abs(np.sort(test1)-np.sort(test2)))
    elif type_comp=="ks_p":
        test1=heat_print[mode].iloc[:,node1]
        test2=heat_print[mode].iloc[:,node2]
        stats=sc.stats.ks_2samp(test1, test2)
        d=1-stats[1]
#print stats[1]

    elif type_comp=="ks_r":
        sorted1=np.sort(heat_print[mode].iloc[:,node1])
        sorted2=np.sort(heat_print[mode].iloc[:,node2])
        sorted3=np.sort(sorted1.tolist()+sorted2.tolist())
        ks=[None]*len(sorted3)
        #print "ne sorted 3:",len(sorted3)
        for i in range(len(sorted3)):
            ks[i]=(len([e for e in sorted1 if e<=sorted3[i]])-len([e for e in sorted2 if e<=sorted3[i]]))*1.0/len(sorted1)
        return np.max(np.abs(ks))  
    else:
        print "comparison type not recognized!!!"
        d=np.nan
    return d

def distances_between_distributions(sig1,sig2,type_comp="auc",mode_diff="agg",normalize="True",plot=False,savefig=False,filefig="plots/nodes_dist.png"):
    ''' Computes the distance between two diffusion patterns (more general form then what was previously proposed)
    '''    
    if type_comp=="auc":
        d=compute_auc(sig1,sig2,normalize=normalize, mode_diff=mode_diff,plot=plot, savefig=savefig,filefig=filefig)
    elif type_comp=="emd":
        ### Required params:
        ### P,Q - Two histograms of size H
        ### D - The HxH matrix of the ground distance between bins of P and Q
        H=30
        hist1,bins_arr=np.histogram(sig1,H)
        #### Normalize histogram
        w=[bins_arr[i+1]-bins_arr[i] for i in range(len(bins_arr)-1)]
        hist1=hist1*1.0/np.matrix(w).dot(hist1)
        hist2,_=np.histogram(sig2,bins_arr)
        hist2=hist2*1.0/np.matrix(w).dot(hist2)
        hist1=np.reshape(np.matrix(hist1), [1, H])
        hist2=np.reshape(np.matrix(hist2), [1, H])
        D=np.zeros((H,H))
        for i in range(H):
            for j in range(H):
                D[i,j]=np.abs(bins_arr[i+1]-bins_arr[j+1])
        d=emd(np.array(hist1.tolist()[0]),np.array(hist2.tolist()[0]),D)
    elif type_comp=="corr":
        d=1-1.0/(np.linalg.norm(sig1))*np.linalg.norm(sig2)*(sig1.dot(sig2))
    elif type_comp=="ks":
        d=np.max(np.abs(np.sort(sig1)-np.sort(sig2)))
    elif type_comp=="ks_p":
        stats=sc.stats.ks_2samp(sig1, sig2)
        d=1-stats[1]
#print stats[1]
    elif type_comp=="ks_r":
        sorted1=np.sort(sig1)
        sorted2=np.sort(sig2)
        sorted3=np.sort(sig1.tolist()+sig2.tolist())
        ks=[None]*len(sorted3)
        #print "ne sorted 3:",len(sorted3)
        for i in range(len(sorted3)):
            ks[i]=(len([e for e in sorted1 if e<=sorted3[i]])-len([e for e in sorted2 if e<=sorted3[i]]))*1.0/len(sorted1)
        d=np.max(np.abs(ks))
    else:
        print "comparison type not recognized!!!"
        d=np.nan
    return d


def test_distances():
    G1,colors1=build_regular_structure(16,"cycle", 4,["fan", 3], start=0,add_random_edges=0,plot=True)
    G2,colors2=build_regular_structure(16,"cycle", 4,["fan", 3], start=0,add_random_edges=0,plot=True)    
    G2.remove_edge(25,26)
    D_12,heat_print1,heat_print2=distances(G1,G2,type_graphs,plot=True,savefig=True)


def distance(list_heat_df,type_comp="auc",normalize=True,mode_diff="agg"):
    '''Computes distance between nodes
    '''
    Distance={}
    N=list_heat_df[0].shape[0]
    for mode in range(len(list_heat_df)):
        dist=np.zeros((N,N))
        for i in range(1,N):
            for j in range(i):
                dist[i,j]=distances_between_nodes(list_heat_df,mode,i,j,type_comp=type_comp,mode_diff="agg",normalize="True")
        Distance[mode]=dist+dist.T
    return Distance


def cross_distances_node(node,list_heat_df,list_heat_df2,mode,type_comp="auc",normalize=True,mode_diff="agg"):
    '''
        Compares nodes across different graphs
        Parameters:
        =================================================================
        list_heat_df,list_heat_df2:       heat_print of graph 1 -- resp 2-- (dictionary: each input corresponds to a different scale)
        type_comp:                        type of distances used to measure similarities between nodes
        normalize:                        only usefulif the type of distance used is AUC
        type_graph:         type of the graphs (should we need to compute the associated heat_print)
        
        
        Returns:
        =================================================================
        d:                 distance
        plots
        '''
    sig1=list_heat_df[mode].iloc[:,node]
    sig2=list_heat_df2[mode].iloc[:,j]
    d=distances_between_distributions(sig1,sig2,type_comp=type_comp,mode_diff=mode_diff,normalize=normalize)
    return d

def cross_distances(list_heat_df,list_heat_df2,type_comp="auc",normalize=True,mode_diff="agg"):
    '''
    Compares nodes across different graphs
    Parameters:
    =================================================================
    list_heat_df,list_heat_df2:       heat_print of graph 1 -- resp 2-- (dictionary: each input corresponds to a different scale)
    type_comp:                        type of distances used to measure similarities between nodes
    normalize:                        only usefulif the type of distance used is AUC
    type_graph:         type of the graphs (should we need to compute the associated heat_print)
    
    
    Returns:
    =================================================================
    d:                 distance
    plots
    '''
    Distance={}
    N=list_heat_df[0].shape[0]
    M=list_heat_df2[0].shape[0]
    for mode in range(len(list_heat_df)):
        dist=np.zeros((N,M))
        for i in range(N):
            sig1=list_heat_df[mode].iloc[:,i]
            for j in range(M):
                sig2=list_heat_df2[mode].iloc[:,j]
                dist[i,j]=distances_between_distributions(sig1,sig2,type_comp=type_comp,mode_diff=mode_diff,normalize=normalize)
        Distance[mode]=dist+dist.T
    return Distance



def compare_graph_chi(rep_G1,rep_G2, taus,t=[],type_comp="global",type_rep="graph",type_graph=["nx","nx"],plot=True, savefig=False,filefig='plots/graph1vs2.pdf'):
    ''' Compares characteristic function representation of the graphs
    
    Parameters:
    =================================================================
    rep_G1,rep_G2:      representations of the graphs (either the graphs, their heat wavelets or the featurized characteristic function)
    taus:               scale parameters to retain (if we want to featurize the representation)
    t:                  sampling points for evaluating the 2D characteristic curve
    type_comp:          type of comparison to achieve: either local (at the node level) or global (graph level)
    type_rep:           type of the representation: can be "graph", "heat_print", or "chi"
    type_graph:         type of the graphs (should we need to compute the associated heat_print)
    
    
    Returns:
    =================================================================
    d:                 distance
    plots 
    
    '''    
    
    
    if type_rep=="graph":
        heat_print1=heat_diffusion(rep_G1,taus=taus,type_graph=type_graph[0],diff_type="heat",b=0)
        heat_print2=heat_diffusion(rep_G2,taus=taus,type_graph=type_graph[1],diff_type="heat",b=0)
        chi1=featurize_characteristic_function(heat_print1,t=t)
        chi2=featurize_characteristic_function(heat_print2,t=t)
    elif type_rep=="heat_print":
        chi1=featurize_characteristic_function(rep_G1,t=t)
        chi2=featurize_characteristic_function(rep_G2,t=t)
    elif type_rep=="chi":
        chi1=rep_G1
        chi2=rep_G2
    else:
        print "type representation is not recognized"
        return False
    N=chi1.shape[0]
    M=chi2.shape[0]
    if type_comp=="global":
        chi1_g=np.mean(chi1,0)
        chi1_g=np.reshape( chi1_g,[len(taus),-1],order='C')
        chi1_g=np.mean(chi1_g,0)
        chi1_g=np.reshape( chi1_g,[-1,2],order='C')
        chi1_g=N*1.0/M*chi1_g
        chi1_g[:,0]+=(M-N)*1.0/M
        chi2_g=np.mean(chi2,0)
        chi2_g=np.reshape( chi2_g,[len(taus),-1],order='C')
        chi2_g=np.mean(chi2_g,0)
        chi2_g=np.reshape( chi2_g,[-1,2],order='C')
        d=np.linalg.norm(chi1_g-chi2_g)
        ax,fig=plt.subplots()
        sb.set_style('white')
        sb.set_context("paper", font_scale=1.5)  
        if plot==True:
            plt.plot(chi1_g[:,0],chi1_g[:,1], c="coral",label="Graph 1")
            plt.plot(chi2_g[:,0],chi2_g[:,1], c="blue",label="Graph 2")
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


def chi_distance_aligned_graph(nodes_names1,nodes_names2,chi1,chi2,nb_t=30):
    intersect=np.intersect1d(nodes_names1,nodes_names2)
    diff1=np.setdiff1d(nodes_names1,nodes_names2)
    diff2=np.setdiff1d(nodes_names2,nodes_names1)
    d=0
    for i in range(len(intersect)):
        ind1=nodes_names1.index(intersect[i])
        ind2=nodes_names2.index(intersect[i])
        d+=np.log(np.linalg.norm(chi1[ind1,:]-chi2[ind2,:]))
    for i in range(len(diff1)):
        ind1=nodes_names1.index(diff1[i])
        d+=np.log(np.linalg.norm(chi1[ind1,:]))
    for i in range(len(diff2)):
        ind2=nodes_names2.index(diff2[i])
        d+=np.log(np.linalg.norm(chi2[ind2,:]))
    return d
    #return 1.0/(chi1.shape[0]+chi2.shape[0]-len(intersect))*d
            
    
