#!/usr/bin/env python

"""
    make-graph.py
"""

from __future__ import division, print_function

import sys
import argparse
import numpy as np
import networkx as nx
from time import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-nodes', type=int, default=3200)
    parser.add_argument('--outpath', type=str, default='synthetic.edgelist')
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    np.random.seed(args.seed)
    
    print("make-graph.py: creating graph", file=sys.stderr)
    W = nx.adjacency_matrix(nx.gnp_random_graph(args.n_nodes, args.p, seed=args.seed + 1))
    keep = np.asarray(W.sum(axis=0) > 0).squeeze()
    W = W[keep]
    W = W[:,keep]
    W.eliminate_zeros()
    
    W = W.tocoo()
    edgelist = np.column_stack([W.row, W.col])
    
    print("make-graph.py: saving graph", file=sys.stderr)
    np.savetxt(args.outpath, edgelist, delimiter='\t', fmt='%d')