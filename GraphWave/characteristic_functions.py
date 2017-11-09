# -*- coding: utf-8 -*-
"""

This file contains the script for defining characteristic functions and using them
as a way to embed distributional information in Euclidean space
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import seaborn as sb
from heat_diffusion import *

def characteristic_function(sig,t,plot=False):
    ''' function for computing the characteristic function associated to a signal at
        a point/ set of points t:
            f(sig,t)=1/len(sig)* [sum_{s in sig} exp(i*t*s)]
    INPUT:
    ===========================================================================
    sig   :      signal over the graph (vector of coefficients)
    t     :      values at which the characteristic function should be evaluated
    plot  :      boolean: should the resulting point/set of points be plotted
    
    OUTPUT:
    ===========================================================================
    f     :      empirical characteristic function
    '''
    f=np.zeros((len(t),3))
    if type(t) is list:
        f=np.zeros((len(t),3))
        f[0,:]=[0,1,0]
        vec1=[np.exp(complex(0,sig[i])) for i in range(len(sig))]
        for tt in range(1,len(t)):
            f[tt,0]=t[tt]
            vec=[x**t[tt] for x in vec1]
            c=np.mean(vec)
            f[tt,1]=c.real
            f[tt,2]=c.imag
        if plot==True:
            plt.figure()
            plt.plot(f[:,1],f[:,2])
            plt.title("characteristic function of the distribution")
    else:
        c=np.mean([np.exp(complex(0,t*sig[i])) for i in range(len(sig))])
        f=[t,np.real(c),np.imag(c)]
    return f


def chi_vary_scale(G,taus=range(1,50,5),t=5,type_graph="nx",plot=True,plot_node=range(0,5,1),savefig=False,filefig="plots/plot_scale.png"):
    ''' function for computing the characteristic function associated to a signal at
        a point/ set of points t:
        f(sig,t)=1/len(sig)* [sum_{s in sig} exp(i*t*s)]
        INPUT:
        ===========================================================================
        G              :    Graph (nx)
        taus           :    list of relevent scales
        t              :    values at which the characteristic function should be evaluated
        plot           :    boolean: should the resulting point/set of points be plotted
        plot_node      :    list of nodes for which we want to plot the characteristic function
        savefig,filefig:    saves the plots to a particular location
        
        OUTPUT:
        ===========================================================================
        phi_s          :    dictionary: each key corresponds to a node, and each associated value
                            is a vector of length 2*len(taus), containing the concatenated real and
                            imaginary values of the characteristic function for the different values of taus
    '''
    heat_print=heat_diffusion(G,taus=taus,type_graph=type_graph)
    phi_s={i:np.zeros(2*len(taus)) for i in range(nx.number_of_nodes(G))}
    for tau in range(len(taus)):
        sig=heat_print[tau]
        for i in range(sig.shape[1]):
            s=sig.iloc[:,i].tolist()
            c=characteristic_function(s,t,plot=False)
            ### Concatenate all the features into one big vector
            phi_s[i][2*tau:2*(tau+1)]= c[1:]
    if plot==True:
        plt.figure()
        cmap = plt.cm.get_cmap('RdYlBu')
        for n in plot_node:
            
            x=[phi_s[n][2*j] for j in range(len(taus))]
            y=[phi_s[n][2*j+1] for j in range(len(taus))]
            plt.scatter(x,y,c=cmap(n),label="node "+ str(n),cmap=cmap)
        
        plt.legend(loc="upper left")
        plt.title("characteristic function of the distribution as s varies")
    return phi_s
        
    
def plot_variation_scale(phi_s,bunch,taus):
    ''' simple function for plotting the variation that is induced
        INPUT:
        ===========================================================================
        phi_s   :    dictionary: each node is a key, and the entries are the concatenated Re/Im values of the characteristic function for the different values in taus (output of chi_vary_scale)
        bunch   :    list of nodes for which to visualize the corresponding  characteristic curves
        taus    :    list of scale values corresponding to phi_s (corresponding input of chi_vary_scale)
        
        OUTPUT:
        ===========================================================================
        None
    '''
    
    plt.figure()
    cmap = plt.cm.get_cmap('RdYlBu')
    for n in bunch:
            
            x=[phi_s[n][2*j] for j in range(len(taus))]
            y=[phi_s[n][2*j+1] for j in range(len(taus))]
            plt.scatter(x,y,c=cmap(n),label="node "+ str(n),cmap=cmap)
        
    plt.legend(loc="upper left")
    plt.title("characteristic function of the distribution as s varies") 
    return
        
def plot_angle_chi(f,t=[], savefig=False, filefig="plots/angle_chi.png"):
    '''Plots the evolution of the angle of a 2D paramteric curve with time
    Parameters
    ----------
    f : 2D paramteric curve (columns corresponds to  X and Y)
    t: (optional) values where the curve is evaluated
    Returns
    -------
    theta: time series of the associated angle (array)
    '''
    
    if len(t)==0:
        t=range(f.shape[0])
    theta=np.zeros(f.shape[0])
    for tt in t:
        theta[tt]=math.atan(f[tt,1]*1.0/f[tt,0])
    return theta
        
def sample_characteristic_function(f):
    return "not implemented yet"



def featurize_characteristic_function_selected_mode(heat_print,mode,t=[]):
    '''uses the heat print as input to produce a feature vector for the distribution for a selected mode
    INPUT:
    ===========================================================================
    heat_print  : dictionary: each key is a mode/scale, and the entry is the dataframe of the diffusion coefficients at each node associated to that particular mode
    mode        : (value  of) the selected scale of interest
    t (optional): evaluation points of the characteristic function
    
    OUTPUT:
    ===========================================================================
    feature vector: pd.DataFrame: each row corresponds to the characteristic function evaluated at point(s) t for a given node
    '''
    if len(t)==0:
        t=range(0,100,5)
        t+=range(85,100)
        t.sort()
        t=np.unique(t)
        t=t.tolist()
    #print len(t)
    chi=np.empty((heat_print[mode].shape[0],2*len(t)))
    sig=heat_print[mode]
    for i in range(sig.shape[1]):
        s=sig.iloc[:,i].tolist()
        c=characteristic_function(s,t,plot=False)
        #print c.shape
            ### Concatenate all the features into one big vector
        chi[i,:]= np.reshape(c[:,1:],[1,2*len(t)])
    #chi=pd.DataFrame(chi, index=sig.columns)
    return chi

def featurize_characteristic_function(heat_print,t=[],nodes=[]):
    ''' same function as above, except the coefficient is computed across all scales and concatenated in the feature vector
    Parameters
    ----------
    heat_print
    t:             (optional) values where the curve is evaluated
    nodes:         (optional at  which nodes should the featurizations be computed (defaults to all)
    
    Returns
    -------
    chi:            feature matrix (pd DataFrame)
    
    '''
    
    if len(t)==0:
        t=range(0,100,5)
        t+=range(85,100)
        t.sort()
        t=np.unique(t)
        t=t.tolist()
    if len(nodes)==0:
        nodes=range(heat_print[0].shape[0])
    chi=np.empty((len(nodes),2*len(t)*len(heat_print)))
    for tau in range(len(heat_print)):
        sig=heat_print[tau]
        for i in range(len(nodes)):
            ind=nodes[i]
            s=sig.iloc[:,ind].tolist()
            c=characteristic_function(s,t,plot=False)
            ### Concatenate all the features into one big vector
            chi[i,tau*2*len(t):(tau+1)*2*len(t)]= np.reshape(c[:,1:],[1,2*len(t)])
    #chi=pd.DataFrame(chi, index=[nodes[i] for i in range(len(nodes))])
    return chi
    
