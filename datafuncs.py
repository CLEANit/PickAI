import torch
import hashlib
import random
from numpy import unique
from random import shuffle

def concat(A):
    l=[]
    for x in A:
        for y in x:
            l.append(y)
    return l

def findinds(x,a):
    ind=x.index(a)
    indl=[ind]
    try:
        while True:
            ind = x[ind+1:].index(a) + ind + 1
            indl.append(ind)
    except ValueError:
        return indl

def str_split(x):
    config_set = unique(x, axis=0)
    return len(config_set)

    #s = str(x)
    #try:
    #    bool(s.index("..."))    #Which is to say that not all the data is plotted
    #    s1 = list(map(str,x))
    #except:
    #    s1=s.strip("tensor([[[[").strip("]]]])").split("]]],\n\n\n        [[[")
    #return s1

def relu(x):
    return (x+abs(x))/2

def nlsl(dum):#net,bth):
    #sl = list(set(dum))
    #sl.sort()
    #nl = [dum.count(x) for x in sl]
    sl, nl = unique(dum, return_counts=True)
    return nl, sl


def hashf(a):
    #try:
    #    ha = str(int(hashlib.md5(a).hexdigest(),16)%4)
    #except:
    ha = int(hashlib.md5(str(a).encode()).hexdigest(),16)
    return ha

def shuf(data):
    #inds = list(range(x))
    sh_inds = list(range(x))
    shuffle(sh_inds)
    #data[inds] = data[sh_inds]
    data[inds] = data[sh_inds]
    return data

def printl(x):
    for xi in x:
        print(xi)

def printr(x):
    for xi in x[::-1]:
        print(xi)

