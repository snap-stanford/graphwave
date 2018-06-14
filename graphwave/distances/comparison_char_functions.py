from graphwave.heat_diffusion import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_heat_diff(node1, node2, mode, list_heat_df, plot=False, savefig=False):
    test1 = np.sort((list_heat_df[mode]).loc[:,node1])
    test2 = np.sort((list_heat_df[mode]).loc[:,node2])
    n = len(list_heat_df[mode])
    ### delta
    w1 = np.array([(test1[i+1] - test1[i]) for i in range(n-1)])
    w2 = np.array([(test2[i+1] - test2[i]) for i in range(n-1)])
    m1 = np.array([0.5 * (test1[i+1] + test1[i]) for i in range(n-1)])
    m2 = np.array([0.5 * (test2[i+1] + test2[i]) for i in range(n-1)])
    #area1=[(delta2[i]-delta1[i])*m1[i]-delta2[i]*test1[i]+test2[i]*delta1[i] for i in range(n-1)]
    area1=[ 0.5 * np.abs(m2[i] - m1[i]) * (w1[i] + w2[i]) for i in range(n-1)]
    if plot==True:
            plt.plot(test1, test2, c="red", label=str(node1)+" vs " +str(node2))
            plt.plot(test1, test1, c="black", label="45 degree line")
            plt.title("qqplot of node "+ str(node1)+ " vs  node " +str(node2))
            plt.legend(loc="upper left")
            plt.xlabel("node "+str(node1))
            plt.ylabel("node "+str(node2))
            if savefig is True:
                plt.savefig("plots/qq_heat_profile"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    
    return np.sum(area1)
