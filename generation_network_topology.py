import torch
import torch.nn as nn
from Hamiltonian import Hamiltonian_torch
nn.requires_grad=False

with torch.no_grad():
    class CNA(nn.Module):
        @torch.no_grad()
        def __init__(self):
            super(CNA, self).__init__()
            self.c1 = nn.Conv2d(1, 16, 3,  stride=1)
            self.c2 = nn.Conv2d(16, 32, 3, stride=1)
            self.d1 = nn.Conv2d(32, 16, 3, stride=1)
            self.d2 = nn.Conv2d(16, 1, 3,  stride=1)
            self.encoder = nn.Sequential(
                    self.c1, 
                    nn.Tanh(),
                    self.c2, 
                    nn.Tanh())
            self.decoder = nn.Sequential(
                    self.d1,
                    nn.Tanh(),
                    self.d2)
    
        @torch.no_grad()
        def one2one(self,x):
            x = torch.sign(2*torch.sign(x) - 1)
            return x

        @torch.no_grad()
        def MakeConfig(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.one2one(x)
            x.squeeze_()
            return x


        @torch.no_grad()
        def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.one2one(x)
            x.squeeze_()
            x = Hamiltonian_torch(x)
            return x
    
