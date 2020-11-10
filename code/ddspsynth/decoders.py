import torch
import torch.nn as nn
import torch.nn.functional as F
from ddspsynth.layers import MLP

class Decoder(nn.Module):
    """Mostly same as the code from DDSP
    Base class to implement any decoder.

    Users should override decode() to define the actual encoder structure.
    Hyper-parameters will be passed through the constructor.
    """

    def __init__(self, param_size):
        super().__init__()
        self.param_size = param_size

    def forward(self, conditioning):
        """
        Outputs Tensor not dictionary
        """
        param_tensor = self.decode(conditioning)
        
        return param_tensor

    def decode(self, conditioning):
        """Takes in conditioning dictionary, returns dictionary of signals."""
        raise NotImplementedError

class DDSPDecoder(Decoder):
    """
    Decoder from DDSP kinda

    Parameters:
        hidden_size (int)       : Size of vectors inside every MLP + GRU + Dense
        param_size (int)        : parameter size of the synthesizer
    """
    def __init__(self, latent_dims, param_size, hidden_size=64, num_layers=1, dropout_p=0.2, use_f0=True, use_ld=True):
        super().__init__(param_size)
        # Map the latent vector
        self.z_MLP  = MLP(latent_dims, hidden_size, 3)
        self.use_f0 = use_f0
        self.use_ld = use_ld
        gru_input_size = hidden_size
        if use_f0:
            self.f0_MLP  = MLP(1, hidden_size, loop=3)
            gru_input_size += hidden_size
        if use_ld:
            self.ld_MLP  = MLP(1, hidden_size, loop=3)
            gru_input_size += hidden_size
        
        # Recurrent model to handle temporality
        self.gru    = nn.GRU(gru_input_size, hidden_size, num_layers, dropout=dropout_p, batch_first=True)
        # Mixing MLP after the GRU
        self.fi_MLP = MLP(hidden_size, hidden_size, loop=3)
        # Outputs to different parameters of the synth
        self.dense_out = nn.Linear(hidden_size, param_size)
        
    def decode(self, conditioning):
        # z: encoder outputs (batch, n_frames, latent_dims)
        outputs = [self.z_MLP(conditioning['z'])]
        if self.use_f0:
            outputs.append(self.f0_MLP(conditioning['f0_scaled']))
        if self.use_ld:
            outputs.append(self.ld_MLP(conditioning['ld_scaled']))
        z = torch.cat(outputs, dim=-1)

        # Recurrent model
        y, _h = self.gru(z) # [batch, n_frames, hidden_size]
        y = self.fi_MLP(y)
        # Retrieve various parameters
        y = self.dense_out(y)  #[batch, n_frames, param_size]
        return y

class MlpDecoder(Decoder):
    """
    Decoder without GRU
    """
    def __init__(self, latent_dims, param_size, hidden_size=256, num_layers=3, use_f0=True, use_ld=True):
        super().__init__(param_size)
        self.z_MLP  = MLP(latent_dims, hidden_size, 3)
        self.use_f0 = use_f0
        self.use_ld = use_ld
        mlp_input_size = hidden_size
        if use_f0:
            self.f0_MLP  = MLP(1, hidden_size, loop=3)
            mlp_input_size += hidden_size
        if use_ld:
            self.ld_MLP  = MLP(1, hidden_size, loop=3)
            mlp_input_size += hidden_size
        # Mixing MLP
        self.fi_MLP = MLP(mlp_input_size, hidden_size, loop=num_layers)
        # Outputs to different parameters of the synth
        self.dense_out = nn.Linear(hidden_size, param_size)

    def decode(self, conditioning):
        # z: encoder outputs (batch, n_frames, latent_dims)
        outputs = [self.z_MLP(conditioning['z'])]
        if self.use_f0:
            outputs.append(self.f0_MLP(conditioning['f0_scaled']))
        if self.use_ld:
            outputs.append(self.ld_MLP(conditioning['ld_scaled']))
        z = torch.cat(outputs, dim=-1)
        
        y = self.fi_MLP(z) # [batch, n_frames, hidden_size]
        # Retrieve various parameters
        y = self.dense_out(y)  #[batch, n_frames, param_size]
        return y