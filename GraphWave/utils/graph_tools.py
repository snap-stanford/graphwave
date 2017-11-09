# -*- coding: utf-8 -*-
"""

Tools for the analysis of the Graph
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sb

def Laplacian(A,norm=False):
    N=A.shape[0]
    D=np.diag([np.sum(A[i,:]) for i in range(N)])
    if norm==True:
        D=np.diag([1.0/np.sqrt(np.sum(A[i,:]) )for i in range(N)])
        L=np.eye(N)-D.dot(A.dot(D))
    else:
        D=np.diag([np.sum(A[i,:]) for i in range(N)])
        L=D-A
    return L
    

def Degree_Matrix(A):
    N=A.shape[0]
    D=np.diag([np.sum(A[i,:]) for i in range(N)])
    return D
    
def InvDegree_Matrix(A):
    N=A.shape[0]
    D_prov=np.array([np.sum(A[i,:]) for i in range(N)])
    ### identify disconnected vertices
    ind_zeros=[i for i in range(N) if D_prov[i]==0]
    np.put(D_prov,ind_zeros,1)
    D_prov[ind_zeros]=1
    D=np.array([1.0/d for d in D_prov])
    np.put(D,ind_zeros,0)
    D=np.diag(D)
    return D


  
def plot_graph(G_nx,f,labels):
    ### plots a graph (of type nx) for the given signal strength f
    pos=nx.spring_layout(G_nx)
    nodes=nx.draw_networkx_nodes(G_nx,pos,node_color=f,cmap="hot", label=labels)
    edges=nx.draw_networkx_edges(G_nx,pos,edge_color="black",width=4)
    labels=nx.draw_networkx_labels(G_nx,pos)
    return True

def normalize_matrix(M, direction="row",type_norm="max"):
    if direction=="row":
        if type_norm=="max":
            D=[1.0/np.max(M[i,:]) for i in range(M.shape[0])]
        elif type_norm=="l2":
            D=[1.0/np.linalg.norm(M[i,:]) for i in range(M.shape[0])]
        elif type_norm=="l1":
            D=[1.0/np.sum(np.abs(M[i,:])) for i in range(M.shape[0])]
        else:
            print "direction not recognized. Defaulting to l2"
            D=[1.0/np.linalg.norm(M[i,:]) for i in range(M.shape[0])]
        D=np.diag(D)
        return D.dot(M)
    elif direction=="column":
        M_tilde=normalize_matrix(M.T, direction="row",type_norm=type_norm)
        return M_tilde.T
    else:
        print "direction not recognized. Defaulting to column"
        return normalize_matrix(M.T, direction="row",type_norm=type_norm)


