import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys
sys.path.append(os.getcwd())
from grad_nettop import CNA
from input_func import *
from smi_func import smi_func #smi_func is the Hamiltonian
from datafuncs import *
import matplotlib.pyplot as plt
import functools as ft
import itertools as itt

import GA_Tools as GA

torch.random.manual_seed(123412352)

try:
    fn = int(sys.argv[1])
except:
    fn = 6


from Hamiltonian import Hamiltonian_multi
def Hist(batch, mod):
    confs = mod.MakeConfig(batch)
    confs = confs.detach().numpy()
    hs = Hamiltonian_multi(confs)
    plt.hist(hs)

def compare_hist(batch, mod1, mod2=None):

    plt.figure()
    Hist(batch, mod1)
    legend = ['mod1']
    
    if mod2 is None:
        blind_sample = np.random.randint(0,2,(1000, fn, fn))
        blind_sample = np.sign(2*np.sign(blind_sample) - 1)
        hs_blind = Hamiltonian_multi(blind_sample)
        plt.hist(hs_blind)

        legend.append("sample")
    
    else:
        #print(type(mod2))
        if type(mod2) == type(""):
            mod2_state = torch.load(mod2)
            mod2 = CNA()
            mod2.load_state_dict(mod2_state)

        
        Hist(batch, mod2)
        legend.append("mod2")
    plt.legend(legend)
    plt.show()


def CalcLoss(outs, ideal_norm):
    #outs_sum = outs[0]
    #for outs_i in outs[1:]:
    #    outs_sum += outs_i

    #ideal_sum = ideal_norm[0]
    #for ideal_i in ideal_norm[1:]:
    #    ideal_sum += ideal_i

    #loss = loss_func(outs_sum, ideal_sum)
    #return loss


    loss_iter = zip(outs, ideal_norm)
    loss_iter = itt.starmap(loss_func, loss_iter)

    # Added-up to preserve gradients
    loss = next(loss_iter)

    for loss_term in loss_iter:
        loss += loss_term

    return loss


import Train_Checkpointing as cp
input_width = cp.DetermineInputSize()    

print("HERE's FN: ",fn)

#
##

#with torch.no_grad():    

loss_func = torch.nn.MSELoss()
#CalcLoss = torch.nn.MSELoss()
#def loss_func(pred, label):
#    loss = (pred - label)**2
#    #loss /= ((pred + label)/2)
#    return loss

from MomentsComparer import MomentsComparer
comparer = MomentsComparer(fn)

batch_size = 5000
batch_mult = 10
epoch_mult = 100

# I checked and GA has a flatter histogram and a lower loss for a given 
# input batch.
mod = CNA()
#GA_trained_state_dict = torch.load('candidate_networks/inet6_1.pt')
#mod.load_state_dict(GA_trained_state_dict)
#mod_ref.load_state_dict(GA_trained_state_dict)

batch_shape = (batch_size, 1, input_width, input_width)
batch = input_func_all(batch_shape, grad_req=True)

comparison_batch = batch
outs_pre = mod(comparison_batch)

comparer.UpdateRange(outs_pre)
smi_range = torch.FloatTensor((comparer.min, comparer.max)) 
norm_fact = torch.max(torch.abs(smi_range))
outs_pre /= norm_fact
    
outs_pre = comparer.EstimateMoments(outs_pre)

ideal = [0.5*torch.ones_like(outs_i) for outs_i in outs_pre]
#ideal = torch.ones_like(outs_pre, requires_grad=True)
#ideal *= 0.123
ideal = torch.FloatTensor(ideal)


learning_rate = 0.0002

opti = torch.optim.SGD(mod.parameters(), lr=learning_rate, momentum=0.0001)
#opti = torch.optim.Adam(mod.parameters(), lr=learning_rate)

# Loss calculated for each moment
loss = CalcLoss(outs_pre, ideal)

loss_avg_fast = loss.item()
loss_avg_slow = loss.item()
avg_decay_fast = 0.1
avg_decay_slow = 0.01
#

# Create single batch
#
mod.train()
for epoch in range(epoch_mult):

    # Zero grad after epoch (This seems artificiall... 
    #  there's no full data-set, only radom inputs)
    
    mod.zero_grad()
    for param in opti.param_groups:
        param['lr'] *= 0.975
    #
    for batch_ind in range(batch_mult):
        batch = input_func_all(batch_shape, grad_req=True)
        
        # Get sample of configuration values
        outs = mod(batch)
        
        # Update comparer targest
        comparer.UpdateRange(outs)
        smi_range = torch.FloatTensor((comparer.min, comparer.max)) 
        norm_fact = torch.max(torch.abs(smi_range))
        outs /= norm_fact
    
        # Estimate moments of configuration-properites sample
        #   > in CNA now
        outs = comparer.EstimateMoments(outs)

        # Get ideal mometns
        ideal = torch.FloatTensor(comparer.ideal_moments)#, requires_grad=True)
        ideal_norm = comparer.CalculateMoments(comparer.max/norm_fact,
                                               comparer.min/norm_fact)
        ideal = torch.FloatTensor(ideal_norm)

        #ideal = torch.ones_like(outs)
        #ideal *= 0.0
        #ideal = [0.5*torch.ones_like(outs_i) for outs_i in outs]
        #ideal = torch.FloatTensor(ideal)
        
        # Loss calculated for each moment
        loss = CalcLoss(outs, ideal)
        loss.backward()
        opti.step()
        #loss_avg = loss_avg * (1-avg_decay) + loss.item() * avg_decay
        loss_avg_fast = loss_avg_fast * (1-avg_decay_fast) + loss.item() * avg_decay_fast
        loss_avg_slow = loss_avg_slow * (1-avg_decay_slow) + loss.item() * avg_decay_slow
        round_digit = 2
        print(round(loss.item(), round_digit),
              round(loss_avg_fast, round_digit),
              round(loss_avg_slow, round_digit)
        )
        #print(loss.item(), loss_avg_fast, loss_avg_slow)

    #net_end = list(mod.parameters())[0].detach().numpy().copy()
    
    #IsSame = (net_start == net_end)
    #IsSame = np.prod(IsSame)
    #print(f"Nets same: {IsSame}")

    
