print("load_make_fungible.py")
import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
sys.path.append(os.getcwd())
from Hamiltonian import Hamiltonian
from generation_network_topology import CNA
from input_func import input_func, input_func_uni
from datafuncs import *
import matplotlib.pyplot as plt

try:
    fn = int(sys.argv[1])
except:
    fn = 6

netl = []
for r in range(1, 10): #Loads several networks.
    net = CNA()
    net.load_state_dict(torch.load("candidate_networks/inet"+str(fn)+"_"+str(r)+".pt"))
    netl.append(net)

data_size = 100000 #mult
standard_in = input_func(data_size, fn, 14)

import itertools as itt
import functools as ft

# Sample producing. Selects against duplicated configurations.
with  h5py.File("ising"+str(fn)+".hdf5",'w') as hf:
    with torch.no_grad():
        grp = hf.create_group("all")
        nind = 0
        
        big_in = input_func(data_size//200,fn)
        big_out = netl[0](big_in)
        big_lab= torch.Tensor([ Hamiltonian(i.squeeze().numpy()) for i in big_out])
        #
        sl,nl = np.unique(big_lab,return_counts=True)
        sl = sl.astype(int).tolist()
        nl = nl.astype(int).tolist()
        for dummi_loop in range(2):
            data_size += data_size*dummi_loop
            while min(nl)<data_size//len(sl): #Recalculated in case new classes appear.
                nind +=1
                little_in = input_func(min(2,nind)*data_size//200,fn,dtype=np.float32)
                little_out = netl[nind%10](little_in)  #Biggest memory?
                little_ham = torch.Tensor([Hamiltonian(i.squeeze().numpy()) for i in little_out])
                #
                sl = list(set( sl + list(set(little_ham.tolist()))))
                sl.sort()
                nl=[len(torch.where(big_lab==x)[0]) for x in sl]
                
                for lab_ind,label in enumerate(sl):     #Over all labels
                    class_size = data_size // len(sl)   #Recaculated in case sl has changed.
                    if nl[sl.index(label)] <  class_size:       #if label_i has fewer than class_size
                        try:
                            new_inds = torch.where(little_ham==label)[0].tolist()
                            new_inds = new_inds[:class_size-nl[lab_ind]]
                            new_data = little_out[new_inds]
                            #
                            new_data = torch.cat((new_data,
                                                  new_data.flip(2),
                                                  new_data.flip(3),
                                                  new_data.flip(2).flip(3),
                                                  new_data.transpose(2,3),
                                                  new_data.transpose(2,3).flip(2),
                                                  new_data.transpose(2,3).flip(3),
                                                  new_data.transpose(2,3).flip(2).flip(3)))
                            # Find duplicates in new_data >> from symmetric data.
                            if sum([int(x<class_size) for x in nl]) > 4:
                                new_data_str = list(map(np.bytes_,new_data.numpy()))
                                #encode each as bytes
                                set_new_data_str = list(set(new_data_str))
                                new_indsp = list(map(new_data_str.index,set_new_data_str))
                                #^ find first instance of each unique datum
                                ind_cuttoff = max(class_size-nl[lab_ind],0)
                                #lind = np.arange(ind_cuttoff)
                                new_indsp = new_indsp[:ind_cuttoff]
                                new_data = new_data[new_indsp] #Select only those.
                            else:
                                #print("len(new_data): ",len(new_data))
                                pass
                            # Shuffle data
                            # shuf. all inds.
                            ndi = np.random.permutation(np.arange(len(new_data)))
                            # select N out of them.
                            ndi = ndi[:max(class_size-nl[lab_ind],0)] 
                            # take only N from new_data.
                            new_data = new_data[ndi] 
                            #
                            big_out = torch.cat((big_out,new_data))                 #
                            #add_lab = [label]*len(new_data) #Since all should have same ham.
                            add_lab = torch.ones(len(new_data))*label
                            big_lab = torch.cat((big_lab,add_lab))
                        except ValueError as e:
                            print(e)
                            pass
                if np.random.random()>0.95: #regular checkin
                    print("nind: ",nind)
                    print("len(big_lab): ",len(big_lab))
                    print("sl: ",sl)
                    print("nl: ",nl)
                    print("~~~~")
            #
        inds = np.arange(len(big_lab))
        shf_inds = np.random.permutation(inds)
        big_out[inds] = big_out[shf_inds]
        big_out = big_out.squeeze()
        big_lab=torch.FloatTensor(big_lab).squeeze()
        big_lab[inds] = big_lab[shf_inds]
        #
        egn=np.random.randint(0,100)
        #
        adict = dict(e=big_lab,c=big_out)
        for k,v in adict.items():
            grp.create_dataset(k,data=v.tolist())
                    
        hf.close()
        big_lab = big_lab.tolist()
        sl = list(set(big_lab))
        sl.sort()
print("Fin")

