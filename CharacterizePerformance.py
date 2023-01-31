print("load_make_fungible.py")
import copy
import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import itertools as itt
import functools as ft

sys.path.append(os.getcwd())
from Hamiltonian import Hamiltonian
from generation_network_topology import CNA
from input_func import input_func, input_func_uni
from datafuncs import *


def rerange(x_og, y):
    '''Linearly scale the range of "x" to that of "y"'''
    x = copy.deepcopy(x_og)
    with torch.no_grad():
        x -= x.min()
        x /= x.max()

        x *= y.max() - y.min()
        x += y.min()
        return x

def SmartSample(data_size, fn, net_index=0, input_width=14):
    with torch.no_grad():
        big_in = input_func(data_size, fn, input_width)
        #big_in = input_func_uni(data_size, fn, input_width)
        big_in = rerange(big_in, standard_in)
        #
        big_out = netl[net_index](big_in)
        big_lab = np.array([out_i.squeeze().numpy() for out_i in big_out])
    return big_lab

def SampleRandomConfigs(data_size, fn):
    big_out = torch.randint(0, 2, (data_size, 1, fn, fn))
    big_out *= 2
    big_out -= 1
    big_lab = np.array([Hamiltonian(i.squeeze().numpy()) for i in big_out])
    return big_lab



if __name__ == "__main__":
    try:
        fn = int(sys.argv[1])
    except:
        fn = 6
    
    netl = []
    for r in range(1, 10): #Loads several networks.
        net = CNA()
        net.load_state_dict(
                torch.load(f"candidate_networks/inet{fn}_{r}.pt")
        )
        netl.append(net)
    
    data_size = 100000 #mult
    standard_in = input_func(data_size, fn,  14)
    
    # Compare top 10 historgrams
    slnl = ft.partial(np.unique, return_counts=True)
    N_sample = 1000
    smarts = (SmartSample(N_sample, fn, net_index=i) for i in range(len(netl)))
    smarts = map(slnl, smarts)
    
    
    # Plot 10 best.
    plt.figure()
    for smart_i in smarts:
        plt.bar(smart_i[0], smart_i[1], align='edge')
    plt.title("Histograms of Top 10 networks")
    plt.xlabel("Derived Quantity Value [Energy]")
    plt.ylabel("Number of Configurations")
    #plt.savefig("top_10_historgrams.png")
    plt.show()
    plt.close()
    
    
    # Compare best network against random sample.
    smart = SmartSample(N_sample, fn)
    smart_sl, smart_nl = np.unique(smart, return_counts=True)
    #
    rand = SampleRandomConfigs(N_sample, fn)
    rand_sl, rand_nl = np.unique(rand, return_counts=True)
    
    
    # Plot histogrmas
    bar_width=0.8
    plt.figure()
    plt.bar(smart_sl, smart_nl, width=bar_width, align='edge')
    plt.bar(rand_sl, rand_nl, width=-bar_width, align='edge')
    plt.legend(["PickAI", "Random"])
    plt.title("PickAI Distribution of Sample Derived Quantities")
    plt.xlabel("Derived Quantity Value")
    plt.ylabel("Number of Occurrences")
    plt.show()
    plt.close()
   
    # Show just random configurations.
    #plt.figure()
    #plt.bar(rand_sl, rand_nl, color='orange', width=4)#, align='edge')
    #plt.title("Internal Energies of 2D Ising Configurations")
    #plt.xlabel("Internal Energies")
    #plt.ylabel("Count")
    #plt.xlim([-(72+4), 72+4])
    #plt.savefig("BestNetwork_vs_RandomSample.png")
    #plt.show()
    
    
