import torch
import torch.nn as nn
import torch.nn.functional as F
from ddspsynth.layers import MLP
from ddspsynth.spectral import loudness_loss

class DDSPSynth(nn.Module):
    """
    DDSP synth with some ae model
    """
    def __init__(self, ae_model, synth):
        super().__init__()
        self.synth = synth
        self.ae_model = ae_model
    
    def encode(self, conditioning):
        conditioning = self.ae_model.encode(conditioning)
        z_tilde, z_loss, z_params = self.ae_model.latent(conditioning)
        conditioning['z'] = z_tilde
        return z_tilde, conditioning
    
    def det_encode(self, conditioning):
        """
        just get the means
        """
        conditioning = self.ae_model.encode(conditioning)
        z_tilde, z_loss, z_params = self.ae_model.latent(conditioning)
        conditioning['z'] = z_params[0] # mean
        return z_params[0], conditioning

    def decode(self, conditioning, n_samples=16000):
        y_params = self.ae_model.decode(conditioning)
        params = self.synth.fill_params(y_params, conditioning)
        resyn_audio, outputs = self.synth(params, n_samples)
        return resyn_audio

    def get_n_frames(self, conditioning, use_ld=False):
        z = self.ae_model.encoder.compute_z(conditioning)
        z_time_steps = z.shape[1]
        f0_time_steps = conditioning['f0_hz'].shape[1]
        if use_ld:
            conditioning = self.ae_model.encoder.fill_loudness(conditioning)
            ld_time_steps = conditioning['ld_scaled'].shape[1]
        else:
            ld_time_steps = 0
        return z_time_steps, f0_time_steps, ld_time_steps

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            [type]: [description]
        """
        y_params, conditioning, z_loss = self.ae_model(conditioning) # batch, L, param_size
        params = self.synth.fill_params(y_params, conditioning)
        resyn_audio, outputs = self.synth(params)
        return resyn_audio

    def train_epoch(self, loader, recon_loss, optimizer, device, beta, clip=1.0, loss_type='L1'):
        # conditions = {'SYNTH PARAM NAME': 'Dataset feature key'} ex.) {'F0':'f0'}
        self.train()
        total_loss = 0
        total_z_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            # Auto-encode
            y_params, conditioning, z_loss = self.ae_model(data_dict) # batch, L, param_size
            params = self.synth.fill_params(y_params, conditioning)
            resyn_audio, outputs = self.synth(params)
            # Reconstruction loss
            batch_loss = recon_loss(data_dict['audio'], resyn_audio, loss_type=loss_type) + beta * z_loss
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            total_loss += batch_loss.detach().cpu()
            total_z_loss += z_loss.detach().cpu()
        total_loss /= len(loader)
        total_z_loss /= len(loader)
        return total_loss, total_z_loss
    
    def eval_epoch(self, loader, recon_loss, device, loss_type='L1'):
        self.eval()
        total_loss = 0
        total_ld_loss = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                # Auto-encode
                resyn_audio = self(data_dict)
                # Reconstruction loss
                batch_loss = recon_loss(data_dict['audio'], resyn_audio, loss_type=loss_type)
                # Perform backward
                total_loss += batch_loss.detach().cpu()
                total_ld_loss += loudness_loss(data_dict['audio'], resyn_audio).detach().cpu()
            total_loss /= len(loader)
            total_ld_loss /= len(loader)
        return total_loss, total_ld_loss