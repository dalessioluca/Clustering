# Classes
import torch

class MLP_x_to_z(torch.nn.Module):
    """ x -> z """
    
    def __init__(self, params):
        super().__init__()
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.z_dim  = params['Z_DIM']
        self.x_dim = self.ch*self.width*self.height
        self.layer1     = torch.nn.Linear(self.x_dim,100, bias=True)
        self.comp_z_mu  = torch.nn.Linear(100,self.z_dim, bias=True)
        self.comp_z_std = torch.nn.Linear(100,self.z_dim, bias=True)
        self.relu       = torch.nn.ReLU()

    def forward(self,x):
        batch_size = x.shape[0]
        x1 = x.view(batch_size,-1)
        x2 = self.relu(self.layer1(x1))
        z_mu  = self.comp_z_mu(x2)
        z_std = torch.exp(self.comp_z_std(x2))
        return z_mu,z_std
    
class MLP_z_to_x(torch.nn.Module):
    """ z -> x """
    
    def __init__(self, params):
        super().__init__()
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.z_dim  = params['Z_DIM']
        self.x_dim = self.ch*self.width*self.height
        self.comp_x_mu  = torch.nn.Linear(self.z_dim,self.x_dim, bias=True)
         
    def forward(self,z):
        batch_size = z.shape[0]
        x_mu = torch.sigmoid(self.comp_x_mu(z)).view(batch_size,self.ch,self.height,self.width)
        return x_mu
        


