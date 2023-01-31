from tqdm import tqdm
import numpy as np
import torch
import h5py

L=4
N=L
def Hamiltonian(A):
    return -np.sum(np.multiply(np.roll(A,1,axis=0),A)+np.multiply(np.roll(A,-1,axis=1),A))
configs = []
energies = []
for i in tqdm(range(2**(L**2))):
    s = np.array([int(_) for _ in list(str(bin(i))[2:].zfill(L**2))]).reshape(L, L)
    s[s==0] = -1
    energy = Hamiltonian(s)
    configs.append(s.tolist())
    energies.append(energy)

energies=torch.FloatTensor(energies)
configs=torch.FloatTensor(configs)

with h5py.File("Ising_again_"+str(L)+".hdf5",'w') as i5:
    grp=i5.create_group("all")
    adict = dict(e=energies,c=configs)
    for k,v in adict.items():
        grp.create_dataset(k,data=v.tolist())

