#### Set of functions to construct a graph as a combination of smaller subgraphs (of a
#### particular shape, defined in the shapes.py file)

import sys
import pygsp
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
from shapes import *


def build_structure(width_basis,basis_type,list_shapes, start=0,add_random_edges=0,plot=False,savefig=False):
	'''This function creates a basis (torus, string, or cycle) and attaches elements of 
	the type in the list randomly along the basis.
	Possibility to add random edges afterwards
	INPUT:
	--------------------------------------------------------------------------------------
	width_basis      :      width (in terms of number of nodes) of the basis
	basis_type       :      (torus, string, or cycle)
	shapes           :      list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)
	start            :      initial nb for the first node
	add_random_edges :      nb of edges to randomly add on the structure
	plot,savefig     :      plotting and saving parameters
	OUTPUT:
	--------------------------------------------------------------------------------------
	Basis            :       a nx graph with the particular shape
	colors           :       labels for each role
	'''
	Basis,index_shape=eval(basis_type)(start,width_basis)
	start+=nx.number_of_nodes(Basis)
	### Sample (with replacement) where to attach the new motives
	plugins=np.random.choice(nx.number_of_nodes(Basis),len(list_shapes), replace=False)
	nb_shape=0
	colors=[0]*nx.number_of_nodes(Basis)
	seen_shapes=["Basis"]
	seen_colors_start=[0]

	for p in plugins:
		index_shape[p]=1
	print index_shape
	col_start=len(np.unique(index_shape))
	for shape in list_shapes:
		shape_type=shape[0]
		col_start=len(np.unique(index_shape)) ## numbers of roles so far
		if shape_type not in seen_shapes:
			print "whoops"
			seen_shapes.append(shape_type)
			seen_colors_start.append(np.max(index_shape)+1)
			col_start=np.max(index_shape)+1
		else:
			ind=seen_shapes.index(shape_type)
			col_start=seen_colors_start[ind]
		args=[start]
		args+=shape[1:]
		args+=[col_start+1]
		S,roles=eval(shape_type)(*args)
		### Attach the shape to the basis
		Basis.add_nodes_from(S.nodes())
		Basis.add_edges_from(S.edges())
		Basis.add_edges_from([(start,plugins[nb_shape])])
		ind=seen_shapes.index(shape_type)
		index_shape[plugins[nb_shape]]+=(-2-ind)
		nb_shape+=1
		colors+=[nb_shape]*nx.number_of_nodes(S)
		index_shape+=roles
		i=seen_shapes.index(shape_type)
		#index_shape+=[2*i]*nx.number_of_nodes(S)
		index_shape[start]=col_start
		start+=nx.number_of_nodes(S)
	print seen_shapes
	if add_random_edges>0:
		## add random edges between nodes:
		for p in range(add_random_edges):
			src,dest=np.random.choice(nx.number_of_nodes(Basis),2, replace=False)
			print src, dest
			Basis.add_edges_from([(src,dest)])
	if plot==True:
		nx.draw_networkx(Basis,node_color=index_shape,cmap="PuRd")
		if savefig==True:
			plt.savefig("plots/structure.png")
	return Basis,colors, plugins,index_shape
	
def build_regular_structure(width_basis,basis_type, nb_shapes,shape, start=0,add_random_edges=0,plot=False,savefig=True):
    ''' This function creates a basis (torus, string, or cycle) and attaches elements of 
    the type in the list regularly along the basis.
    Possibility to add random edges afterwards
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)
	start            :      initial nb for the first node
	add_random_edges :      nb of edges to randomly add on the structure
	plot,savefig     :      plotting and saving parameters
    OUTPUT:
    --------------------------------------------------------------------------------------
    Basis            :       a nx graph with the particular shape
    colors           :       labels for each role
    '''
    
    Basis,_=eval(basis_type)(start,width_basis)
    start+=nx.number_of_nodes(Basis)
    ### Sample (with replacement) where to attach the new motives
    K=math.floor(width_basis/nb_shapes)
    plugins=[k*K for k in range(nb_shapes)]
    nb_shape=0
    colors=[1 if index in plugins else 0 for index in range(nx.number_of_nodes(Basis)) ]
    col_start=len(np.unique(colors))
    for s in range(nb_shapes):
        type_shape=shape[0]
        args=[start]
        if len(shape)>1:
            args+=shape[1:]
        args+=[col_start+1]
        S,roles_shape=eval(type_shape)(*args)
        ### Attach the shape to the basis
        Basis.add_nodes_from(S.nodes())
        Basis.add_edges_from(S.edges())
        Basis.add_edges_from([(start,plugins[nb_shape])])
        #colors+=[3]*nx.number_of_nodes(S)
        colors+=roles_shape
        colors[start]-=1
        start+=nx.number_of_nodes(S)
        nb_shape+=1
    if add_random_edges>0:
        ## add random edges between nodes:
        for p in range(add_random_edges):
            src,dest=np.random.choice(nx.number_of_nodes(Basis),2, replace=False)
            print src, dest
            Basis.add_edges_from([(src,dest)])
    if plot==True:
        nx.draw_networkx(Basis,pos=nx.layout.fruchterman_reingold_layout(Basis),node_color=colors,cmap="PuRd")
        if savefig==True:
            plt.savefig("plots/regular_structure.png")
    return Basis,colors









def build_lego_structure(list_shapes, start=0,betweenness_density=2.5,plot=False,savefig=False,save2text=''):
    '''This function creates a graph from a list of building blocks by adding edges between blocks.
    INPUT:
    ---------------------------------------------------------------------------------
    list_shapes           :   list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)
    betweenness_density   :   density of the backbone structure
    start                 :   initial node nb 
    plot, savefig,save2txt:   plotting and saving parameters
   
    OUTPUT:
    ---------------------------------------------------------------------------------
    G:                    :   a nx graph (associationg of cliques/motifs planted along backbone structure)
    colors                :   motif nb (per individual motif)
    index_roles           :   role id
    label_shape           :   label/class of the motif
    '''
    
    G=nx.Graph()
    
    nb_shape=0
    colors=[]         ## labels for the different shapes
    seen_shapes=[]
    seen_colors_start=[]  ## pointer for where should the next shape's labels be initialized
    index_roles=[]  ### roles in the network
    col_start=0
    label_shape=[]
    for shape in list_shapes:
        shape_type=shape[0]
        if shape_type not in seen_shapes:
            seen_shapes.append(shape_type)
            seen_colors_start.append(np.max([0]+index_roles)+1)
            col_start=seen_colors_start[-1]
            ind=len(seen_colors_start)-1
        else:
            ind=seen_shapes.index(shape_type)
            col_start=seen_colors_start[ind]
        start=len(index_roles)
        args=[start]
        args+=shape[1:]
        args+=[col_start]
        S,roles=eval(shape_type)(*args)
        ### Attach the shape to the basis
        G.add_nodes_from(S.nodes())
        G.add_edges_from(S.edges())
        nb_shape+=1
        colors+=[nb_shape]*nx.number_of_nodes(S)
        index_roles+=roles
        label_shape+=[col_start]*nx.number_of_nodes(S)
    print seen_shapes
    ### Now we link the different shapes:
    N=G.number_of_nodes()
    A=np.ones((N,N))
    np.fill_diagonal(A,0)
    for j in np.unique(colors):
        ll=np.array([e==j for e in colors])
        A[ll,:][:,ll]=0
    ### Randomly select edges to put between shapes:
        n=pymc.distributions.rtruncated_poisson(betweenness_density,1)[0]
        start_k=np.array(range(N))[np.array(ll)]
        idx, idy=np.nonzero(A[ll,:])
        indices=np.random.choice(range(len(idx)),n)
        G.add_edges_from([(1+start_k[0]+idx[i],1+idy[i]) for i in indices])

    if plot==True:
        nx.draw_networkx(G,node_color=index_roles,cmap="PuRd")
        if savefig==True:
            plt.savefig("plots/structure.png")
    if len(save2text)>0:
        graph_list_rep=[['Id','shape_id','type_shape','role']]+\
        [[i+1,colors[i],label_shape[i],index_roles[i]] for i in range(nx.number_of_nodes(G))]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        elist=[['Source','Target']]+[[e[0],e[1]] for e in G.edges()]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        np.savetxt(save2text+"graph_edges.txt",elist,fmt='%s,%s')
    return G,colors, index_roles, label_shape
    
    
def build_lego_structure_from_structure(list_shapes, start=0,plot=False,savefig=False,graph_type='nx.connected_watts_strogatz_graph', graph_args=[4,0.4],save2text='',add_node=10):
    '''same as before, except that the shapes are put on top of the spanning tree of a graph
    instead of random edges
     INPUT:
    ---------------------------------------------------------------------------------
    list_shapes           :   list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)
    graph_type            :   which type of backbone graph (default= 'nx.connected_watts_strogatz_graph')
    add_nodes             :   number of "empty nodes" to add to the graph structures, ie, nodes in the graph that do not belong to a particular clique
    graph_args            :   arguments for generating the backbone graph (except from nb of nodes)
    start                 :   initial node nb 
    plot, savefig,save2txt:   plotting and saving parameters
   
    OUTPUT:
    ---------------------------------------------------------------------------------
    G:                    :   a nx graph (associationg of cliques/motifs planted along backbone structure)
    colors                :   motif nb (per individual motif)
    index_roles           :   role id
    label_shape           :   label/class of the motif
    '''
    G=nx.Graph()
    
    nb_shape=0
    colors=[]         ## labels for the different shapes
    seen_shapes=[]
    seen_colors_start=[]  ## pointer for where should the next shape's labels be initialized
    index_roles=[]  ### roles in the network
    col_start=0
    label_shape=[]
    
    for shape in list_shapes:
        shape_type=shape[0]
        if shape_type not in seen_shapes:
            seen_shapes.append(shape_type)
            seen_colors_start.append(np.max([0]+index_roles)+1)
            col_start=seen_colors_start[-1]
            ind=len(seen_colors_start)-1
        else:
            ind=seen_shapes.index(shape_type)
            col_start=seen_colors_start[ind]
        start=len(index_roles)
        print 'start=',start
        args=[start]
        args+=shape[1:]
        args+=[col_start]
        S,roles=eval(shape_type)(*args)
        ### Attach the shape to the basis
        G.add_nodes_from(S.nodes())
        G.add_edges_from(S.edges())
        
        colors+=[nb_shape]*nx.number_of_nodes(S)
        index_roles+=roles
        label_shape+=[col_start]*nx.number_of_nodes(S)
        nb_shape+=1
    #print seen_shapes
    ### Now we link the different shapes:
    N=G.number_of_nodes()
    N_prime=nb_shape
    #### generate Graph
    graph_args.insert(0,N_prime+add_node)
    G.add_nodes_from(range(N,N+add_node))
    colors+=[nb_shape+rr for rr in range(add_node)]
    #print colors
    r=np.max(index_roles)+1
    l=label_shape[-1]
    index_roles+=[r]*add_node
    label_shape+=[-1]*add_node
    Gg=eval(graph_type)(*graph_args)
    elist=[]
    ### permute the colors:
    initial_col=np.unique(colors)
    perm=np.unique(colors)
    np.random.shuffle(perm)
    color_perm={initial_col[i]:perm[i] for i in range(len(np.unique(colors)))}
    colors2=[color_perm[c] for c in colors]
    #colors=colors2
    for e in Gg.edges():
        if e not in elist:
            ii=np.random.choice(np.where(np.array(colors2)==(e[0]))[0],1)[0]
            jj=np.random.choice(np.where(np.array(colors2)==(e[1]))[0],1)[0]
            G.add_edges_from([(ii,jj)])
            elist+=[e]
            elist+=[(e[1],e[0])]

    if plot==True:
        nx.draw_networkx(G,node_color=index_roles,cmap="PuRd")
        if savefig==True:
            plt.savefig("plots/structure.png")
    if len(save2text)>0:
        graph_list_rep=[['Id','shape_id','type_shape','role']]+\
        [[i,colors[i],label_shape[i],index_roles[i]] for i in range(nx.number_of_nodes(G))]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        elist=[['Source','Target']]+[[e[0],e[1]] for e in G.edges()]
        np.savetxt(save2text+"graph_nodes.txt",graph_list_rep,fmt='%s,%s,%s,%s')
        np.savetxt(save2text+"graph_edges.txt",elist,fmt='%s,%s')
    return G,colors, index_roles, label_shape
    
