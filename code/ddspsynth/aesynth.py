import torch
import torch.nn as nn

from ddspsynth.aes import AE, VAE


class AESynth(AE):
    """
    an autoencoding model with a differentiable synthesizer after the decoder network
    """
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, synth):
        super(AESynth, self).__init__()
        self.synth = synth

    def decode(self, z, cond=None):
        z = self.decoder(z)
        if cond:
            x_tilde = self.synth(z, cond)
        else:
            x_tilde = self.synth(z)
        return x_tilde
    
    def train_epoch(self, loader, loss, optimizer, device, beta):
        self.train()
        full_loss = 0
        for (x, f0, loud, y) in loader:
            # Send to device
            x, f0, loud = [it.to(device, non_blocking=True) for it in [x, f0, loud]]
            f0, loud = f0.transpose(1, 2), loud.transpose(1, 2)
            # Auto-encode
            x_tilde, z_tilde, z_loss = self((x, (f0, loud)))
            # Reconstruction loss
            rec_loss = loss(x_tilde, y) / float(x.shape[1] * x.shape[2])
            # Final loss
            b_loss = (rec_loss + (beta * z_loss)).mean(dim=0)
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss, device):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for (x, f0, loud, y) in loader:
                # Send to device
                x, f0, loud = [it.to(device, non_blocking=True) for it in [x, f0, loud]]
                f0, loud = f0.transpose(1, 2), loud.transpose(1, 2)
                # Auto-encode
                x_tilde, z_tilde, z_loss = self((x, (f0, loud)))
                # Final loss
                rec_loss = loss(x_tilde, y)
                full_loss += rec_loss
            full_loss /= len(loader)
        return full_loss

class VAESynth(AESynth, VAE):
    """
    VAE version of AESynth
    """
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, synth):
        AESynth.__init__(self, encoder, decoder, encoder_dims, latent_dims, synth)
        VAE.__init__(self, encoder, decoder, encoder_dims, latent_dims)
