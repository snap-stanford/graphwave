# -*- coding: utf-8 -*-
"""
This file contains the script for defining characteristic functions
and using them as a way to embed distributional information
in Euclidean space
"""
import cmath
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def characteristic_function(sig, time_pnts, plot=False):
    '''
    function for computing the characteristic function
    associated to a signal at a point/ set of points t:
    $$   f(sig,t)=1/len(sig)* [sum_{s in sig} exp(i*t*s)] $$
    INPUT:
    ===========================================================================
    sig           :      signal over the graph (vector of coefficients)
    time_pnts     :      values at which the char. function should be evaluated
    plot          :      boolean: should the resulting set
                         of points be plotted?

    OUTPUT:
    ===========================================================================
    f             :      empirical characteristic function. Array of
                         size of len(t) * 3 (col1: t, col2: Re[phi(t)],
                         col3: Im[phi(t)])
    '''
    time_pnts = list(time_pnts)
    n_time_pnts = len(time_pnts)
    f = np.zeros((n_time_pnts, 3))
    f[0, :] = [0, 1, 0]
    vec1 = np.array([np.exp(complex(0, sig[i])) for i in range(len(sig))])
    for tt in range(1, n_time_pnts):
        f[tt, 0] = time_pnts[tt]
        vec = [x**time_pnts[tt] for x in vec1]
        c = np.mean(vec)
        f[tt, 1] = c.real
        f[tt, 2] = c.imag
    if plot is True:
        plt.figure()
        plt.plot(f[:, 1], f[:, 2])
        plt.title('characteristic function of the distribution')
    return f


def plot_characteristic_function(phi_s, bunch, time_pnts, ind_tau):
    ''' simple function for plotting the variation that is induced
        INPUT:
        ===========================================================================
        phi_s   :    array: each node is a row,
                     and the entries are the concatenated Re/Im values of
                     the characteristic function for the different
                     values in taus (output of chi_vary_scale)
        bunch   :    list of nodes for which to visualize the corresponding
                     characteristic curves
        taus    :    list of scale values corresponding to phi_s
                     (corresponding input of chi_vary_scale)

        OUTPUT:
        ===========================================================================
        None
    '''
    sb.set_style('white')
    plt.figure()
    n_time_pnts = len(time_pnts)
    cmap = plt.cm.get_cmap('RdYlBu')
    for n in bunch:
            x = [phi_s[n, ind_tau * n_time_pnts + 2 * j]
                 for j in range(n_time_pnts)]
            y = [phi_s[n, ind_tau * n_time_pnts + 2 * j + 1]
                 for j in range(n_time_pnts)]
            plt.scatter(x, y, c=cmap(n), label="node "+str(n), cmap=cmap)
    plt.legend(loc='upper left')
    plt.title('characteristic function of the distribution as s varies')
    plt.show()
    return


def plot_angle_chi(f, t=[], savefig=False, filefig='plots/angle_chi.png'):
    '''Plots the evolution of the angle of a 2D paramteric curve with time
    Parameters
    ----------
    f : 2D paramteric curve (columns corresponds to  X and Y)
    t: (optional) values where the curve is evaluated
    Returns
    -------
    theta: time series of the associated angle (array)
    '''
    if len(t) == 0:
        t = range(f.shape[0])
    theta = np.zeros(f.shape[0])
    for tt in t:
        theta[tt] = math.atan(f[tt, 1] * 1.0 / f[tt, 0])
    return theta


def featurize_characteristic_function(heat_print, t=[], nodes=[]):
    '''
    same function as above, except the coefficient is computed across
    all scales and concatenated in the feature vector
    Parameters
    ----------
    heat_print
    t          :         (optional) values where the curve is evaluated
    nodes      :         (optional at  which nodes should the featurizations
                         be computed (defaults to all)

    Returns
    -------
    chi        :            feature matrix (pd DataFrame)
    '''
    n_filters, n_nodes, _ = heat_print.shape
    if len(t) == 0:
        t = range(0, 100, 5)
        t += range(85, 100)
        t.sort()
        t = np.unique(t)
        t = t.tolist()
    n_time_pnts = len(t)
    if len(nodes) == 0:
        nodes = range(n_nodes)
    chi = np.empty((n_nodes, 2 * n_time_pnts * n_filters))
    for tau in range(n_filters):
        sig = heat_print[tau, :, :]
        for i in range(len(nodes)):
            ind = nodes[i]
            s = sig[:, ind].tolist()
            c = characteristic_function(s, t, plot=False)
            # Concatenate all the features into one big vector
            index_update = range(tau * 2 * n_time_pnts,
                                 (tau + 1) * 2 * n_time_pnts)
            chi[i, index_update] = np.reshape(c[:, 1:], [1, 2 * n_time_pnts])
    return chi
