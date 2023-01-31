#import numpy as np
from torch import randn, FloatTensor, float32, rand
def input_func(data_size, fn, input_width=None, dtype=float32):
    if input_width is None:
        input_width = fn
    #return torch.FloatTensor(torch.randint(-4, 5, (data_size,1,fn,fn), dtype=torch.float32))
    #return torch.FloatTensor(torch.randn((data_size,1,fn,fn),dtype=torch.float32))
    return FloatTensor(randn((data_size,1,input_width,input_width),dtype=float32))

def input_func_all(data_dims,dtype=float32, grad_req=False):
    #return torch.FloatTensor(torch.randint(-4,5,data_dims,dtype=torch.float32))

    return FloatTensor(randn(data_dims,
                       dtype=float32,
                       requires_grad=grad_req))

def input_func_uni(data_size, fn, input_width=None, dtype=float32):
    if input_width is None:
        input_width = fn
    return FloatTensor(rand((data_size,1,input_width,input_width),dtype=float32))

