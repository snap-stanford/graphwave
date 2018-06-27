# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:32:21 2018

@author: cdonnat
"""

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from graphwave.shapes import build_graph
from graphwave.graphwave import *
import time
np.random.seed(123)
import copy
import math

def compute_cheb_coeff(scale, order):
    coeffs = [(-scale)**k * 1.0 / math.factorial(k) for k in range(order + 1)]
    return coeffs

def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2*i-1)*1.0/(2*order) * math.pi) for i in range(1,order+1)])
    basis = [np.ones((1,order)),np.array(xx)]
    for k in range(order + 1-2):
        basis.append(2* np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale*(xx+1))
    products = np.einsum("j,ij->ij",f,basis)
    coeffs = 2.0/(order)*products.sum(1)
    coeffs[0] = coeffs[0]/2
    return list(coeffs)




def heat_diffusion_ind(graph, taus=[1, 10, 25, 50], diff_type='heat',order = 10, proc = 'approximate'):
    '''
    This method computes the heat diffusion waves for each of the nodes
    INPUT:
    -----------------------
    graph    :    Graph, can be of type networkx or pygsp
    taus     :    list of 4 scales for the wavelets. The higher the tau,
                  the better the spread
    order    :    order of the polynomial approximation

    OUTPUT:
    -----------------------
    heat     :     tensor of length  len(tau) x n_nodes x n_nodes
                   where heat[tau,:,u] is the wavelet for node u
                   at scale tau
    '''
    # Compute Laplacian

    a = nx.adjacency_matrix(graph)
    n_nodes, _ = a.shape
    thres = np.vectorize(lambda x : x if x > 0.001*1.0/n_nodes else 0)
    lap = laplacian(a)
    n_filters = len(taus)
    if proc == 'exact':
        lamb, U = np.linalg.eigh(lap.todense())
        heat = {}
        for i in range(n_filters):
             heat[i] = U.dot(np.diagflat(np.exp(-taus[i]*lamb).flatten())).dot(U.T)
    else:
        heat = {i: sc.sparse.csc_matrix((n_nodes, n_nodes)) for i in range(n_filters) }
        #monome = {0: sc.sparse.eye(n_nodes)}
        #for k in range(1, order + 1):
        #     monome[k] = lap.dot(monome[k-1])
        monome = {0: sc.sparse.eye(n_nodes), 1: lap - sc.sparse.eye(n_nodes)}
        for k in range(2, order + 1):
             monome[k] = 2 * (lap - sc.sparse.eye(n_nodes)).dot(monome[k-1]) - monome[k - 2]
        for i in range(n_filters):
            coeffs = compute_cheb_coeff_basis(taus[i], order)
            #print(coeffs)
            heat[i] = sc.sum([  coeffs[k] * monome[k]  for k in range(0, order + 1)])
            index = heat[i].nonzero
            temp = thres(heat[i].A)
            heat[i] = sc.sparse.csc_matrix(temp)
            #### trim the data:

             #for k in range(0, order + 1):
                 #heat[i] +=  coeffs[k] * monome[k]
    return heat, taus



import copy

def charac_function(time_points, temp):
#time_points= np.linspace(0,100,101)
#if True
    

    complexify = np.vectorize( lambda x: np.exp(-np.complex(0,x)))
    n_nodes = temp.shape[1]
    sig = np.zeros((len(time_points), n_nodes), dtype='complex128')
    nnz_vec = np.array([1.0/n_nodes*(temp[:,i].nnz) for i in range(n_nodes)])
    temp2 = copy.deepcopy(temp)
    for it_t, t in enumerate(time_points):
        temp2.data = complexify(t*temp.data)
        sig[it_t,:] = 1.0/n_nodes *(temp2.sum(0)) + nnz_vec
    
    final_sig = np.zeros((2*sig.shape[0],sig.shape[1]))
    final_sig[::2,:] = np.real(sig)
    final_sig[1::2,:]= np.imag(sig)
    return final_sig

def charac_function_multiscale(heat, time_points):
    final_sig = []
    for i in heat.keys():
        final_sig.append(charac_function(time_points, heat[i]))
    return np.vstack(final_sig).T


def laplacian(a):
        n_nodes, _ = a.shape
        posinv = np.vectorize(lambda x: 1.0/np.sqrt(x) if x>1e-10 else 1)
        d = sc.sparse.diags(np.array(posinv(a.sum(0))).reshape([-1,]),0)
        lap = sc.sparse.eye(n_nodes) - d.dot(a.dot(d))
        return lap

def graphwave2(graph, taus, time_pnts, type_graph="nx",
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
    if taus == 'auto':
        a = nx.adjacency_matrix(G)
        lap = laplacian(a)
        l1 = sc.sparse.linalg.eigsh(lap, 2,  which='SM',return_eigenvectors=False)[0]
        smax = -np.log(0.75) * np.sqrt( 0.5 / l1)
        smin = -np.log(0.99) * np.sqrt( 0.5 / l1)
        taus = np.linspace(smin, smax, 4)
    heat_print, _ = heat_diffusion_ind(graph, list(taus), diff_type='heat', order=50, proc = 'approximate')
    chi = charac_function_multiscale(heat_print, time_pnts)

    return chi


if True:
    width_basis = 50
    nbTrials = 20


    ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
    ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

    ### 1. Choose the basis (cycle, torus or chain)
    basis_type = "cycle"

    ### 2. Add the shapes
    n_shapes = 30 ## numbers of shapes to add
    #shape=["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
    #shape=["star",6]
    list_shapes = [["house"]] * n_shapes

    ### 3. Give a name to the graph
    identifier = 'AA'  ## just a name to distinguish between different trials
    name_graph = 'houses'+ identifier
    sb.set_style('white')

    ### 4. Pass all these parameters to the Graph Structure
    add_edges = 0
    G, communities, _ , role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                           add_random_edges=add_edges, plot=False,
                                           savefig=False)
    tic =time.time()
    time_pts = list(np.arange(0,10,0.5)) + list(np.arange(10,110,10))
    chi = graphwave2(G, 'auto', time_pts)
    toc =time.time()
    print toc-tic

