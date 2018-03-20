#!/usr/bin/env python


"""
    heat.py
"""


from __future__ import division, print_function

import sys
import numpy as np
from scipy import sparse
from time import time

try:
    import cupy
except:
    print('WARNING: install cupy > 2.0 for GPU support', file=sys.stderr)

# --
# Helpers

def estimate_lmax(L):
    try:
        lmax = sparse.linalg.eigsh(L, k=1, tol=5e-3, ncv=min(L.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1.01
        return lmax
    
    except sparse.linalg.ArpackNoConvergence:
        return 2


def compute_laplacian(W):
    dw = np.asarray(W.sum(axis=1)).squeeze()
    d = np.power(dw, -0.5)
    D = sparse.diags(np.ravel(d), 0).tocsc()
    return sparse.identity(W.shape[0]) - D * W * D


def characteristic_function(s, t=np.arange(0, 100, 2)):
    return (np.exp(np.complex(0, 1) * s) ** t.reshape(-1, 1)).mean(axis=1)

# --
# Heat kernel

class Heat(object):
    
    def __init__(self, W=None, taus=None, L=None, lmax=None):
        assert isinstance(taus, list)
        
        if L is None:
            self.num_nodes = W.shape[0]
            self.L= compute_laplacian(W)
        else:
            self.num_nodes = L.shape[0]
            self.L = L
        
        self.lmax = estimate_lmax(self.L) if lmax is None else lmax
        self.taus = taus
        
    def _compute_cheby_coeff(self, tau, order=30):
        N = order + 1
        a = self.lmax / 2.
        
        tmpN = np.arange(N)
        num  = np.cos(np.pi * (tmpN + 0.5) / N)
        
        c = np.zeros(N)
        for o in range(N):
            kernel = lambda x: np.exp(-tau * x / self.lmax)
            c[o] = 2. / N * np.dot(kernel(a * num + a), np.cos(np.pi * o * (tmpN + 0.5) / N))
            
        return c
    
    def _filter(self, signal, order=30):
        assert signal.shape[0] == self.num_nodes
        n_signals = signal.shape[1]
        n_features_out = len(self.taus)
        
        c = [self._compute_cheby_coeff(tau=tau, order=order) for tau in self.taus]
        c = np.atleast_2d(c)
        
        heat_print = np.zeros((n_features_out, self.num_nodes, n_signals))
        a = self.lmax / 2.
        
        twf_old = signal
        twf_cur = 1. / a * (self.L.dot(signal) - a * signal)
        
        for i in range(n_features_out):
            tmp = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur
            if sparse.issparse(tmp):
                tmp = tmp.todense()
            
            heat_print[i] = tmp
        
        factor = 2 / a * (self.L - a * sparse.eye(self.num_nodes))
        for k in range(2, c.shape[1]):
            twf_new = factor.dot(twf_cur) - twf_old
            
            for i in range(n_features_out):
                tmp = c[i, k] * twf_new
                if sparse.issparse(tmp):
                    tmp = tmp.todense()
                
                heat_print[i] += tmp
            
            twf_old = twf_cur
            twf_cur = twf_new
        
        return heat_print.transpose((0, 2, 1))
    
    def featurize(self, signal, order=30):
        heat_print = self._filter(signal, order=order)
        
        feats = []
        for i, sig in enumerate(heat_print):
            sig_feats = []
            for node_sig in sig:
                node_feats = characteristic_function(node_sig)
                node_feats = np.column_stack([node_feats.real, node_feats.imag]).reshape(-1)
                sig_feats.append(node_feats)
            
            feats.append(np.vstack(sig_feats))
        
        return np.hstack(feats)


class CupyHeat(Heat):
    
    def __init__(self, W=None, taus=None, L=None, lmax=None):
        super(CupyHeat, self).__init__(W=W, taus=taus, L=L, lmax=lmax)
        self.L = cupy.sparse.csr_matrix(self.L)
    
    def _filter(self, signal, order=30):
        signal = cupy.array(signal)
        assert signal.shape[0] == self.num_nodes, 'signal.shape[0] != self.num_nodes'
        n_signals = signal.shape[1]
        n_features_out = len(self.taus)
        
        c = [self._compute_cheby_coeff(tau=tau, order=order) for tau in self.taus]
        c = np.atleast_2d(c)
        
        heat_print = cupy.zeros((n_features_out, self.num_nodes, n_signals))
        a = self.lmax / 2.
        
        twf_old = signal
        twf_cur = 1. / a * (self.L.dot(signal) - a * signal)
        
        tmpN = np.arange(self.num_nodes, dtype=int)
        for i in range(n_features_out):
            heat_print[i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur
        
        factor = 2 / a * (self.L - a * cupy.sparse.eye(self.num_nodes))
        for k in range(2, c.shape[1]):
            twf_new = factor.dot(twf_cur) - twf_old
            
            for i in range(n_features_out):
                heat_print[i] += c[i, k] * twf_new
            
            twf_old = twf_cur
            twf_cur = twf_new
        
        return heat_print.get()

