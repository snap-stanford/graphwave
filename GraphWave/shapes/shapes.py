# -*- coding: utf-8 -*-
"""

"""

#### Set of functions to construct shapes in a network(i.e, subgraphs of a
#### particular shape)


import pygsp
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sc

def test_graph(plot=False,savefig=False):
        
    G=nx.Graph()
    G.add_nodes_from(range(30))
    colors=["grey"]
    colors+=["red"]*7
    G.add_edges_from([(1,2),(1,3),(1,4),(2,4),(2,3),(4,5),(5,6),(6,7),(7,3),(7,4)])
    colors+=["grey"]*3
    G.add_edges_from([(11,12),(11,13),(11,14),(12,14),(12,13),(14,15),(15,16),(16,17),(17,13),(17,14)])
    colors+=["blue"]*7
    colors+=["grey"]*3
    G.add_edges_from([(21,22),(21,23),(21,24),(22,24),(22,23),(24,25),(25,26),(26,27),(27,23),(27,24)])
    colors+=["green"]*7
    colors+=["grey"]*3    
    G.add_edges_from([(7,8),(8,9),(9,10),(9,29),(8,30),(30,20),(20,8),(20,22),(22,18),(18,19),(18,12),(12,9),(18,8)])
    G.add_edges_from([(0,10),(29,0),(0,28)])
    G.add_edges_from([(19,31),(31,33),(19,32)])
    colors+=["orange"]*3
    if plot==True:
        nx.draw_networkx(G)
    if savefig==True:
        plt.savefig("Graph_test1.png")
    return G,colors
    
    
def diamond(start,col_start=0,plot=False):
    #### INPUT:
    #### -------------
    #### start: "starting index" for the shape

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    G.add_nodes_from(range(start,start+6))
    G.add_edges_from([(start,start+1),(start+1,start+2),(start+2,start+3),(start+3,start)])
    G.add_edges_from([(start+4,start),(start+4,start+1),(start+4,start+2),(start+4,start+3)])
    G.add_edges_from([(start+5,start),(start+5,start+1),(start+5,start+2),(start+5,start+3)])
    if plot==True:
        nx.draw_networkx(G)
        plt.savefig("plots/diamond.png")
    colors=[col_start]*6
    return G,colors

def cycle(start,len_cycle,col_start=0,plot=False):
    #### INPUT:
    #### -------------
    #### start: "starting index" for the shape
    #### len_cycle: length of the cycle

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    G.add_nodes_from(range(start,start+len_cycle))
    for i in range(len_cycle-1):
        G.add_edges_from([(start+i,start+i+1)])
    G.add_edges_from([(start+len_cycle-1,start)])
    if plot==True:
        nx.draw_networkx(G)
        plt.savefig("plots/cycle.png")
    return G,[col_start]*len_cycle


def house(start,col_start=0,plot=False):
    #### INPUT:
    #### -------------
    #### start: "starting index" for the shape

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    G.add_nodes_from(range(start,start+5))
    G.add_edges_from([(start,start+1),(start+1,start+2),(start+2,start+3),(start+3,start)])
    G.add_edges_from([(start,start+2),(start+1,start+3)])
    G.add_edges_from([(start+4,start),(start+4,start+1)])
    colors=[col_start,col_start,col_start+1,col_start+1,col_start+2]
    if plot==True:
        nx.draw_networkx(G,node_color=colors)
        plt.savefig("plots/house.png")
    return G,colors

def string(start, width,col_start=0):
    G=nx.Graph()
    G.add_nodes_from(range(start,start+width))
    for i in range(width-1):
         G.add_edges_from([(start+i,start+i+1)])
    colors=[col_start]*width
    colors[0]=col_start+1
    colors[-1]=col_start+1
    return G,colors
    


def star(start,nb_branches,col_start=0, plot=False):
    
    #### INPUT:
    #### -------------
    #### start: "starting index" for the shape
    #### nb_branches: nb of branches for the star

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    G.add_nodes_from(range(start,start+nb_branches+1))
    for k in range(1,nb_branches+1):
        G.add_edges_from([(start,start+k)])
    if plot==True:
        nx.draw_networkx(G,node_color=colors,cmap="hot")
        plt.savefig("plots/star.png")
    colors=[col_start+1]*(nb_branches+1)
    colors[0]=col_start
    return G,colors

def fan(start,nb_branches,col_start=0, plot=False):
    G,colors=star(start,nb_branches,col_start=col_start)
    for k in range(1,nb_branches-1):
        colors[k]+=1
        colors[k+1]+=1
        G.add_edges_from([(start+k,start+k+1)])
    if plot==True:
        nx.draw_networkx(G,node_color=colors,cmap="hot")
        plt.savefig("plots/fan.png")
    return G,colors

def clique(start, nb_nodes,nb_to_remove=0,col_start=0,plot=False):
    ### Defines a clique (complete graph on nb_nodes nodes, with nb_to_remove  edges that will have to be removed)
    A=np.ones((nb_nodes, nb_nodes))
    np.fill_diagonal(A,0)
    G=nx.Graph()
    #G.add_nodes_from(range(start,start+nb_nodes+1))
    G=nx.from_numpy_matrix(A)
    edge_list=G.edges()
    
    colors=[col_start]*nb_nodes
    if nb_to_remove>0:
        lst=np.random.choice(len(edge_list),nb_to_remove, replace=False)
        to_delete=[edge_list[e] for e in lst]
        G.remove_edges_from(to_delete)
        for e in lst:
            print edge_list[e][0]
            print len(colors)
            colors[edge_list[e][0]]+=1
            colors[edge_list[e][1]]+=1
    mapping={k:(k+start) for k in range(nb_nodes)}
    G=nx.relabel_nodes(G,mapping)
    if plot==True:
        nx.draw_networkx(G,node_color=colors,cmap="hot")
    return G,colors

def tree(start,nb_levels,regularity,col_start=0, plot=False):
    #### INPUT:
    #### -------------
    #### start: "starting index" for the shape
    #### nb_levels: nb of levels in the tree
    #### regularity: nb of children for each node

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G=nx.Graph()
    nodes_level=[regularity**l for l in range(nb_levels) ]
    G.add_nodes_from(range(start,start+np.sum(nodes_level)))
    a=start
    it=0
    for n in range(1,np.sum(nodes_level)):
        G.add_edges_from([(a,start+n)])
        it+=1
        if it==(regularity):
            a+=1
            it=0
            
    N=nx.number_of_nodes(G)
    colors=[col_start+1]*nx.number_of_nodes(G)
    colors[0]=col_start
    for i in range(regularity**l):
        colors[N-1-i]+=1
    if plot==True:
        nx.draw_networkx(G,node_color=colors,cmap="hot")
        plt.savefig("plots/tree.png")
    return G,colors
    

def hollow(start,col_start=0,plot=False):
    ### Creates a torus-like basis structure
        #### INPUT:
    #### -------------

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    G1,_=cycle(start,5)
    G,_=cycle(start+5,10)
    G.add_nodes_from(G1.nodes())
    G.add_edges_from(G1.edges())
    G.add_edges_from([(start,start+5),(start+1,start+7),(start+2,start+9),(start+3,start+11),(start+4,start+13)])
    G.add_edges_from([(start+6,start+1),(start+6,start)])
    G.add_edges_from([(start+8,start+2),(start+8,start+1)])
    G.add_edges_from([(start+10,start+3),(start+10,start+2)])
    G.add_edges_from([(start+12,start+4),(start+12,start+3)])
    G.add_edges_from([(start+14,start),(start+14,start+4)])
    if plot==True:
        nx.draw_networkx(G,node_color=colors,cmap="hot")
        plt.savefig("plots/hollow.png")
    return G, [col_start]*nx.number_of_nodes(G)


def create_graph_combination(plot=False):
    ### Creates one instance of a graph
    G=hollow(0)
    start=G.number_of_nodes()
    colors=["black"]*start
    G1,_=house(start)
    G.add_nodes_from(G1.nodes())
    G.add_edges_from(G1.edges())
    G.add_edges_from([(0,start)])
    colors+=["red"]*G1.number_of_nodes()
    start+=G1.number_of_nodes()
    G2,_=house(start)
    G.add_nodes_from(G2.nodes())
    G.add_edges_from(G2.edges())
    G.add_edges_from([(12,start)])
    start+=G2.number_of_nodes()
    colors+=["blue"]*G2.number_of_nodes()
    G3,_=house(start)
    G.add_nodes_from(G3.nodes())
    G.add_edges_from(G3.edges())
    G.add_edges_from([(8,start)])
    start+=G3.number_of_nodes()
    colors+=["green"]*G3.number_of_nodes()
    F,_=fan(start,6)
    G.add_nodes_from(F.nodes())
    G.add_edges_from(F.edges())
    start+=F.number_of_nodes()
    G.add_edges_from([(14,start)])

    #start+=F.number_of_nodes()
    colors+=["orange"]*F.number_of_nodes()
    F2,_=fan(start,3)
    G.add_nodes_from(F2.nodes())
    G.add_edges_from(F2.edges())
    G.add_edges_from([(start-1,start)])
    colors+=["yellow"]*F2.number_of_nodes()
    start+=F2.number_of_nodes()
    
    if plot==True:
        nx.draw_networkx(G)
    return G,colors
        
    
    
def department(start,nb_levels):
    ### Tries to recreate the structure of a department
    G,_=tree(start,nb_levels,2)
    ### add secretrary
    sec=start+G.number_of_nodes()
    G.add_node(sec)
    G.add_edges_from([(sec,start),(sec,start+1),(sec,start+3), (sec,start+5),(sec,start+2), (sec,start+6)])
    G.add_node(sec+1)
    G.add_edges_from([(sec+1,sec-1)])
    
    
def graph_with_cells(plot):
    ### Another attempt at a graph
    G,_=cycle(0,6)
    H1,_=house(start)
    H2,_=house(start)
    H3,_=house(start)
    C,_=cycle(start,4)
    C2,_=cycle(start,4)
    C3,_=cycle(start,4)



def type_shapes():
    ### Returns a dictionary mapping the shapes to their id number
    dict_shape={1:"department",2:"tree",3:"Fan"}
    return dict_shape
    
def build_structure(width_basis,basis_type,list_shapes, start=0,add_random_edges=0,plot=False,savefig=False):
    ### This function creates a basis (torus, string, or cycle) and attaches randomly elements of the type in the list
    ### Possibility to add random edges afterwards
    #### INPUT:
    #### -------------
    #### width_basis: width (in terms of number of nodes) of the basis
    #### basis_type: (torus, string, or cycle)
    #### list_shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    Basis,index_shape=eval(basis_type)(start,width_basis)
    start+=nx.number_of_nodes(Basis)
    ### Sample (with replacement) where to attach the new motives
    plugins=np.random.choice(nx.number_of_nodes(Basis),len(list_shapes), replace=False)
    #plugins=range(0,nx.number_of_nodes(Basis),nx.number_of_nodes(Basis)//len(list_shapes))
    nb_shape=0
    colors=[0]*nx.number_of_nodes(Basis)
    #index_shape=[0]*nx.number_of_nodes(Basis)
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
    ### This function creates a basis (torus, string, or cycle) and attaches randomly elements of the type in the list
    ### Possibility to add random edges afterwards
    #### INPUT:
    #### -------------
    #### width_basis: width (in terms of number of nodes) of the basis
    #### basis_type: (torus, string, or cycle)
    #### shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

    #### OUTPUT:
    #### -------------
    #### Basis: a nx graph with the particular shape:
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

       

def create_bigger_network(nb_cells, width_cell,list_shapes,cell_type="cycle"):
    #### Automatically creates a big network
    G,colors,plugins=build_structure(width_basis,basis_type,list_shapes, start=0,add_random_edges=0,plot=False)
    start=G.number_of_nodes()    
    for i in range(1, nb_cells):
        Gi,colors_i,plugins_i=build_structure(width_basis,basis_type,list_shapes, start=start,add_random_edges=0,plot=False)
        G.add_nodes_from(Gi.nodes())
        G.add_edges_from(Gi.edges())
        G.add_edges_from([(start,start+1)])
        start+=Gi.number_of_nodes() 
        colors+=colors_i
        plugins+=plugins_i
    return G,colors,plugins
    


def barbel_graph(start,N1, N2,plot=False,savefig=False):
    ### Creates a Barbell-Graph (two dense components connected by a string)
        #### INPUT:
    #### -------------
    #### start: "starting index" for the graph
    #### N1: nb of nodes in each cluster
    #### 2*N2+1: nb of links 

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    
    A1=np.ones((N1,N1))
    np.fill_diagonal(A1,0) ### set the diagonal to 0
    G=nx.from_numpy_matrix(A1)
    A2=np.ones((N1,N1))
    C=np.zeros((N1,N1))
    A=np.bmat([[A1,C],[C,A2]])
    np.fill_diagonal(A,0) ### set the diagonal to 0
    G=nx.from_numpy_matrix(A)
    start=nx.number_of_nodes(G)
    G.add_nodes_from(range(start,start+2*N2+1))
    string=[(i,i+1) for i in range(start,start+2*N2) ]
    G.add_edges_from(string)
    G.add_edges_from([(0,start),(N1,start+2*N2)])
    index_shape=[0]*(2*N1)
    index_shape[0]=1
    index_shape[N1]=1
    str_role=[i for i in range(2, N2+2)]
    str_role.append(N2+2)
    str_role+=[N2+3-i for i in range(2, N2+2) ]
    index_shape+=str_role
    if plot==True:
        nx.draw_networkx(G,pos=nx.layout.fruchterman_reingold_layout(G),node_color=index_shape,cmap="hot")
        if savefig==True:
            plt.savefig("plots/barbel_structure.png")
    return G,index_shape


def karate_club(plot=False,savefig=False):
    ## Defines the mirrored- Karate network structure that was used in the KDD paper
    G1=nx.karate_club_graph()
    A=nx.adjacency_matrix(G1)
    N=nx.number_of_nodes(G1)
    B=np.zeros((N,N))
    A=np.bmat([[A.todense(),B],[B,A.todense()]])
    G=nx.from_numpy_matrix(A)
    ## add link between two random members
    index_shape=range(N)
    index_shape+=range(N)
    G.add_edges_from([(0,36)])
    if plot==True:
        nx.draw_networkx(G,pos=nx.layout.fruchterman_reingold_layout(G),node_color=index_shape,cmap="hot")
        if savefig==True:
            plt.savefig("plots/karate_structure.png")
    return G,index_shape

    
def mirrored_cavemen(n,k,plot=False,savefig=False):
    ## Mirrored-caveman graph
    N=n*k
    A=np.ones((k,k))
    np.fill_diagonal(A,0)
    Adj=sc.linalg.block_diag(*([A]*n))
    for i in range(n-1):
        #Adj[i*k,(i+1)*k]=1
        #Adj[(i+1)*k,i*k]=1
        Adj[(i+1)*k-1,(i+1)*k]=1
        Adj[(i+1)*k,(i+1)*k-1]=1
    Adj[n*k-1,0]=1
    Adj[0,n*k-1]=1
    G=nx.from_numpy_matrix(Adj)    
    index_shape=[0]* N
    for i in range(n-1):
        index_shape[(i+1)*k-1]=1
        index_shape[(i+1)*k]=1
    index_shape[0]=1
    index_shape[n*k-1]=1
    if plot==True:
        #plt.figure()
        #plt.axis("off")
        nx.draw_networkx(G,pos=nx.layout.fruchterman_reingold_layout(G),node_color=index_shape,cmap="hot")
        if savefig==True:
            plt.savefig("plots/connected_cavemen.png")
    return G,index_shape
    
   
def build_lego_structure(list_shapes, start=0,betweenness_density=2.5,plot=False,savefig=False,save2text=''):
    ### This function creates a graph from a list of building blocks by addiing edges between blocks
    #### INPUT:
    #### -------------
    #### width_basis: width (in terms of number of nodes) of the basis
    #### basis_type: (torus, string, or cycle)
    #### list_shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
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
    ### same as before, except that the shapes are put on top of the spanning tree of a graph
    ### This function creates a graph from a list of building blocks by addiing edges between blocks
    #### INPUT:
    #### -------------
    #### width_basis: width (in terms of number of nodes) of the basis
    #### basis_type: (torus, string, or cycle)
    #### list_shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
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
    
    
 
