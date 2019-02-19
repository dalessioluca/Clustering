# Classes
import torch
import pyro

class MLP_x_to_p_dz(torch.nn.Module):
    """ x -> p,dz """
    
    def __init__(self, params):
        super().__init__()
        self.K      = params['K']
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.z_dim  = params['Z_DIM']
        self.x_dim = self.ch*self.width*self.height
   
        self.comp_p = torch.nn.Linear(self.x_dim,self.K, bias=True)
        self.comp_dz_mu = torch.nn.Linear(self.x_dim,self.z_dim, bias=True)
        self.comp_dz_std = torch.nn.Linear(self.x_dim,self.z_dim, bias=True)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()

    def forward(self,x):
        batch_size = x.shape[0]
        x1 = x.view(batch_size,-1)
        p = self.softmax(self.comp_p(x1))
        dz_mu= self.tanh(self.comp_dz_mu(x1))
        dz_std= torch.exp(self.comp_dz_std(x1))
        return p,dz_mu,dz_std
    
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

class VAE_no_latent_structure(torch.nn.Module):
    
    def __init__(self,params,encoder,decoder):
        super().__init__()
        
        # Parameters
        self.use_cuda = params['use_cuda']
        self.ch     = params['CHANNELS']
        self.width  = params['WIDTH']
        self.height = params['HEIGHT']
        self.z_dim  = params['Z_DIM']
        self.x_dim = self.ch*self.width*self.height
        
        # Instantiate the encoder and decoder
        self.decoder = decoder
        self.encoder = encoder
        
        if(self.use_cuda):
            self.cuda()
        
    def guide(self, imgs=None):
        """ 1. run the inference 
            2. sample latent variables 
        """       
        #-----------------------#
        #--------  Trick -------#
        #-----------------------#
        if(imgs is None):
            observed = False
            imgs = torch.zeros(8,self.ch,self.height,self.width)
            if(self.use_cuda):
                imgs=imgs.cuda()
        else:
            observed = True
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#

        batch_size,ch,width,height = imgs.shape
        pyro.module("encoder", self.encoder)
        z_mu,z_std = self.encoder(imgs)
        with pyro.plate('batch_size', batch_size, dim=-1):
            z = pyro.sample('z_latent', dist.Normal(z_mu,z_std).to_event(1))
        return z_mu,z_std
            
    def model(self, imgs=None):
        """ 1. sample the latent from the prior:
            2. runs the generative model
            3. score the generative model against actual data 
        """
        #-----------------------#
        #--------  Trick -------#
        #-----------------------#
        if(imgs is None):
            observed = False
            imgs = torch.zeros(8,self.ch,self.height,self.width)
            if(self.use_cuda):
                imgs=imgs.cuda()
        else:
            observed = True
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#
        sigma = pyro.param("sigma", 0.01*imgs.new_ones(1))
        batch_size,ch,width,height = imgs.shape
        pyro.module("decoder", self.decoder)
        with pyro.plate('batch_size', batch_size, dim=-1):
            z = pyro.sample('z_latent', dist.Normal(imgs.new_zeros(self.z_dim),10*imgs.new_ones(self.z_dim)).to_event(1))
            x_mu = self.decoder(z) #x_mu is between 0 and 1
            pyro.sample('obs', dist.Normal(x_mu.view(batch_size,-1),sigma).to_event(1), obs=imgs.view(batch_size,-1))
        return x_mu
    
    def reconstruct(self,imgs):
        z_mu,z_std = self.encoder(imgs)
        x = self.decoder(z_mu)
        return x


