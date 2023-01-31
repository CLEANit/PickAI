import torch
import torch.nn as nn
from Hamiltonian import Hamiltonian_torch
from MomentsComparer import MomentsComparer
class CNA(nn.Module):
    def __init__(self):
        super(CNA, self).__init__()
        self.c1 = nn.Conv2d(1, 16, 3,  stride=1)#, padding=1) #6/8 - 6/6
        #self.c2 = nn.Conv2d(16, 32, 3, stride=1)#, padding=1) #6/8 - 7/7
        #self.d1 = nn.Conv2d(32, 16, 3, stride=1)#, padding=1) #7/9 - 
        self.d2 = nn.Conv2d(16, 1, 3,  stride=1)#,padding=1)
        self.encoder = nn.Sequential(
                self.c1, 
                nn.Tanh(),
                #self.c2, 
                #nn.Tanh()
        )
        self.decoder = nn.Sequential(
                #self.d1,
                #nn.Tanh(),
                self.d2,
                nn.Softsign()
        )

    def one2one(self,x):
        x = torch.sign(2*torch.sign(x) - 1)
        #x = torch.sign(x)
        #x = torch.nn.Softsign()(x)
        return x

    #@torch.no_grad()
    #def Hamiltonian_torch_bkQM(A):
    #    t1 = torch.roll(A, 1, dims=1)
    #    t1 = torch.multiply(t1, A)
    #
    #    t2 = torch.roll(A, -1, dims=2)
    #    t2 = torch.multiply(t2, A)
    #    t = torch.add(t1, t2)
    #    t = -torch.sum(t, (1,2))
    #    return t

    def MakeConfig(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #x = self.one2one(x)
        x.squeeze_()
        return x


    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        #x = self.one2one(x)
        x.squeeze_()
        x = Hamiltonian_torch(x)
        #x = MomentsComparer.EstimateMoments(x)
        return x

