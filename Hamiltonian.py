import numpy as np
import torch
def Hamiltonian(A):
    #A = np.copy(AA)
    #A = np.reshape(arr,(int(np.sqrt(arr.shape[0])),int(np.sqrt(arr.shape[0]))))
    return -np.sum(np.multiply(np.roll(A,1,axis=0),A)+np.multiply(np.roll(A,-1,axis=1),A))

def Hamiltonian_multi(A):
    t1 = np.roll(A, 1, axis=1)
    t1 = np.multiply(t1, A)

    t2 = np.roll(A, -1, axis=2)
    t2 = np.multiply(t2, A)
    t = np.add(t1, t2)
    t = -np.sum(t, axis=(1,2))
    return t


def Hamiltonian_torch(A):
    t1 = torch.roll(A, 1, dims=1)
    t1 = torch.multiply(t1, A)

    t2 = torch.roll(A, -1, dims=2)
    t2 = torch.multiply(t2, A)
    t = torch.add(t1, t2)
    t = -torch.sum(t, (1,2))
    return t
N = 10
fn = 6
a = torch.randn((N,1,fn,fn), requires_grad=True)
b = torch.sign(2*torch.sign(a) - 1)
bn = b.detach().numpy()
ct = Hamiltonian_torch(b)
cn = Hamiltonian_multi(bn)
bm = bn.reshape((-1, fn, fn))
cm = map(Hamiltonian, bm)
cm = np.array(list(cm))

#def hexp(A):
#    t1 = np.roll(A,1,axis=0)
#    t1 = np.multiply(t1, A)
#
#    t2 = np.roll(A,-1,axis=1)
#    t2 = np.multiply(t2,A)
#
#    t = t1+t2
#    s = -np.sum(t)
#    return s
#
#from functools import partial
#from numpy import roll, multiply
#from numpy import sum as np_sum
#
#def roll_mult(A, direction, axis_dim):
#    term = roll(A, direction, axis=axis_dim)
#    term = multiply(term, A)
#    return term
#
#def gen(N=3, n=5):
#    a = np.random.randint(0,2,(N,n,n))
#    a *=2
#    a -= 1
#    return a
#
#from itertools import product, starmap
#def twoterms_attempt2(A):
#    #terms = product(A, (-1, 1)) #product enters into A...
#    terms = ((A, direction, axis_dim) for axis_dim, direction in enumerate((1, -1)))
#    terms = starmap(roll_mult, terms)
#
#    term_sum = sum(terms)
#    term_sum = -np.sum(term_sum)
#    #terms = np.sum(A, axis=0) #axis=0??
#
#
#    return term_sum
#
#A = gen()
#a = A[0]
#print(f"Hamiltonian(a) :{Hamiltonian(a)}")
