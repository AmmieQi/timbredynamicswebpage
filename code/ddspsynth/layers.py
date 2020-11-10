import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Copied from pytorch-DDSP
    Implementation of the MLP, as described in the original paper

    Parameters :
        in_size (int)   : input size of the MLP
        out_size (int)  : output size of the MLP
        loop (int)      : number of repetition of Linear-Norm-ReLU
    """
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            )] + [nn.Sequential(nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ) for i in range(loop - 1)])

    def forward(self, x):
        for lin in self.linear:
            x = lin(x)
        return x

class FiLM(nn.Module):
    """
    feature-wise linear modulation
    """
    def __init__(self, input_dim, attribute_dim):
        super().__init__()
        self.input_dim = input_dim
        self.generator = nn.Linear(attribute_dim, input_dim*2)
        
    def forward(self, x, c):
        """
        x: (*, input_dim)
        c: (*, attribute_dim)
        """
        c = self.generator(c)
        gamma = c[..., :self.input_dim]
        beta = c[..., self.input_dim:]
        return x*gamma + beta

class FiLMMLP(nn.Module):
    """
    MLP with FiLMs in between
    """
    def __init__(self, in_size, out_size, attribute_dim, loop=3):
        super().__init__()
        self.loop = loop
        self.mlps = nn.ModuleList([nn.Linear(in_size, out_size)] 
                                + [nn.Linear(out_size, out_size) for i in range(loop-1)])
        self.films = nn.ModuleList([FiLM(out_size, attribute_dim) for i in range(loop)])

    def forward(self, x, c):
        """
        x: (*, input_dim)
        c: (*, attribute_dim)
        """
        for i in range(self.loop):
            x = self.mlps[i](x)
            x = F.relu(x)
            x = self.films[i](x, c)
        return x

class Normalize1d(nn.Module):
    """
    normalize over the last dimension
    ddsp normalizes over time dimension of mfcc
    """
    def __init__(self, channels, norm_type='instance', batch_dims=1):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(channels, affine=True)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(channels, affine=True)
        self.flat = nn.Flatten(0, batch_dims-1)

    def forward(self, x):
        """
        First b_dim dimensions are batch dimensions
        Last dim is normalized
        """
        orig_shape = x.shape
        x = self.flat(x)
        if len(x.shape) == 2:
            # no channel dimension
            x = x.unsqueeze(1)
        x = self.norm(x)
        x = x.view(orig_shape)
        return x

class Normalize2d(nn.Module):
    """
    take the average over 2 dimensions (time, frequency)
    """
    def __init__(self, norm_type='instance'):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(1)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        """
        3D input first of which is batch dim
        [batch, dim1, dim2]
        """
        x = self.norm(x.unsqueeze(1)).squeeze(1) # dummy channel
        return x