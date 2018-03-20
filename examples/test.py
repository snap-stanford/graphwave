#!/usr/bin/env python

"""
    test.py
"""

from __future__ import division, print_function

import pygsp
import numpy as np
import networkx as nx
from time import time

import heat
from helpers import par_graphwave

# --
# Create graph

np.random.seed(123)

num_nodes = 1000
W = nx.adjacency_matrix(nx.gnp_random_graph(num_nodes, 0.01, seed=123 + 1))
W.eliminate_zeros()

taus = [0.5, 0.6, 0.7]
s = np.eye(num_nodes)

# --
# Simple test

t = time()
heat_kernel = heat.Heat(W=W, taus=taus)
feats = heat_kernel.featurize(s)
print("heat.Heat: took %fs" % (time() - t))


# --
# Parallel test

t = time()
par_feats = par_graphwave(heat_kernel, n_chunks=2, n_jobs=2, verbose=10)
print("par_graphwave: took %fs" % (time() - t))

assert np.allclose(feats, par_feats)


# --
# Cupy test

t = time()
cupy_feats = heat.CupyHeat(W=W, lmax=heat_kernel.lmax, taus=taus).featurize(s)
print("heat.CupyHeat: took %fs" % (time() - t))

assert np.allclose(feats, cupy_feats)
assert np.allclose(par_feats, cupy_feats)

