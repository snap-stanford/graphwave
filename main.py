#!/usr/bin/env python

"""
    main.py
"""

from __future__ import division, print_function

import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from time import time
from scipy import sparse
from scipy.sparse import lil_matrix as li
import heat
from helpers import par_graphwave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, required=False)
    parser.add_argument('--outpath', type=str, required=False)
    parser.add_argument('--taus', type=str, default="0.5")
    parser.add_argument('--n-queries', type=int, default=-1)
    parser.add_argument('--n-chunks', type=int, default=1)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    np.random.seed(args.seed)
    taus = map(float, args.taus.split(','))
    
    print("main.py: loading %s" % args.inpath, file=sys.stderr)
    
    edgelist = np.array(pd.read_csv(args.inpath, sep=' ', header=None, dtype=int))
    print("before W:")
    print(np.arange(edgelist.shape[0]))
    print(edgelist[:,0])
    print(edgelist[:,1])
    lll = edgelist.max()+1
    W = li((lll,lll),dtype=int)
    for i in range(len(edgelist)):
        v = edgelist[i]
        W[v[0],v[1]]=1
    W = W.tocsr()
    # W = sparse.csr_matrix((np.arange(edgelist.shape[0]), (edgelist[:,0], edgelist[:,1])))
    # print(W.shape())
    W = ((W + W.T) > 0).astype(int)
    
    if not (np.sum(W, axis=0) > 0).all():
        print("main.py: dropping isolated nodes", file=sys.stderr)
        keep = np.asarray(W.sum(axis=0) > 0).squeeze()
        print("keep:",len(keep))
        W = W[keep]
        W = W[:,keep]
    
    print("main.py: running on graph w/ %d edges" % W.nnz, file=sys.stderr)
    
    t = time()
    hk = heat.Heat(W=W, taus=taus)
    
    if args.n_queries < 0:
        args.n_queries = hk.num_nodes
    
    print("main.py: running %d queries" % args.n_queries, file=sys.stderr)
    pfeats = par_graphwave(hk, n_queries=args.n_queries, n_chunks=args.n_chunks, n_jobs=args.n_jobs, verbose=10)
    
    run_time = time() - t
    print("main.py: took %f seconds -- saving %s.npy" % (run_time, args.outpath), file=sys.stderr)
    
    np.save(args.outpath, pfeats)