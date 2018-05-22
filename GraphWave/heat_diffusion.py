# -*- coding: utf-8 -*-
"""
All functions for computing the wavelet distributions
"""
import networkx as nx
import pygsp
import numpy as np


def heat_diffusion(graph, taus=[1, 10, 25, 50], diff_type='heat',
                   type_graph='pygsp'):
    '''
    This method computes the heat diffusion waves for each of the nodes
    INPUT:
    -----------------------
    graph    :    Graph, can be of type networkx or pygsp
    taus     :    list of 4 scales for the wavelets. The higher the tau,
                  the better the spread
    type     :    type of the  graph (networkx or pygsp)

    OUTPUT:
    -----------------------
    heat     :     tensor of length  len(tau) x n_nodes x n_nodes
                   where heat[tau,:,u] is the wavelet for node u
                   at scale tau
    '''
    if type_graph == 'pygsp':
        pygsp_graph = graph
        n_nodes = graph.N
        if pygsp_graph.lap_type != 'normalized':
            pygsp_graph.create_laplacian('normalized')
    elif type_graph == 'nx':
        adj = nx.adjacency_matrix(graph)
        n_nodes = graph.number_of_nodes()
        pygsp_graph = pygsp.graphs.Graph(adj, lap_type='normalized')
    else:
        print 'graph type not recognized'
        return None

    # Call the appropriate filter
    n_filters = len(taus)

    if diff_type == 'mexican':
        n_filters = 6
        Hk = pygsp.filters.MexicanHat(pygsp_graph, n_filters)
    elif diff_type == 'wave':
        Hk = pygsp.filters.Wave(pygsp_graph, taus, normalize=True)
    else:
        Hk = pygsp.filters.Heat(pygsp_graph, taus, normalize=False)

    heat = np.zeros((len(taus), n_nodes, n_nodes))

    Sf_vec = Hk.analysis(np.eye(n_nodes))
    for i in range(n_filters):
        heat[i, :, :] = Sf_vec[i * n_nodes: (i + 1) * n_nodes, :]     # stores in tensor the results

    return heat, taus
