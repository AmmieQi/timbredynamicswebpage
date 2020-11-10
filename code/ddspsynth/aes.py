import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence
from ddspsynth.layers import MLP

class AE(nn.Module):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.map_latent = nn.Linear(encoder_dims, latent_dims)
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.01, 0.01)
            m.bias.data.fill_(0.01)     
        
    def encode(self, conditioning):
        conditioning = self.encoder(conditioning)
        return conditioning
    
    def decode(self, conditioning):
        return self.decoder(conditioning)
    
    def latent(self, conditioning):
        z_enc = conditioning['z']
        z = self.map_latent(z_enc)
        #latent and latent loss
        return z, torch.zeros(1).to(z.device), [z, torch.zeros_like(z)]

    def forward(self, conditioning):
        # Encode the inputs
        conditioning = self.encode(conditioning)
        # Potential regularization
        z_tilde, z_loss, z_params = self.latent(conditioning)
        # write actual z
        conditioning['z'] = z_tilde
        # Decode the samples to get synthesis parameters
        y_params = self.decode(conditioning)
        return y_params, conditioning, z_loss

class VAE(AE):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super(VAE, self).__init__(encoder, decoder, encoder_dims, latent_dims)
        # Latent gaussians
        self.mu = nn.Linear(encoder_dims, latent_dims)
        self.log_var = nn.Linear(encoder_dims, latent_dims)
        self.apply(self.init_parameters)
    
    def latent(self, conditioning):
        z_enc = conditioning['z']
        batch_size, num_frames, _ = z_enc.shape
        mu = self.mu(z_enc)
        log_var = self.log_var(z_enc)
        eps = torch.randn_like(mu).detach().to(mu.device)
        z = log_var.exp().sqrt() * eps + mu
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / (batch_size*num_frames)
        return z, kl_div, [mu, log_var.exp().sqrt()]

class VAE2(AE):
    #just for testing
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__(encoder, decoder, encoder_dims, latent_dims)
        # Latent gaussians
        self.linear_mu = nn.Sequential(
                nn.Linear(encoder_dims, latent_dims),
                nn.ReLU(),
                nn.Linear(latent_dims, latent_dims))
        self.linear_logvar = nn.Sequential(
                nn.Linear(encoder_dims, latent_dims),
                nn.ReLU(),
                nn.Linear(latent_dims, latent_dims))

        # self.mu = nn.Linear(encoder_dims, latent_dims)
        # self.log_var = nn.Linear(encoder_dims, latent_dims)
        self.apply(self.init_parameters)
    
    def latent(self, conditioning):
        z_enc = conditioning['z']
        batch_size, num_frames, _ = z_enc.shape
        mu = self.linear_mu(z_enc)
        log_var = self.linear_logvar(z_enc)
        eps = torch.randn_like(mu).detach().to(mu.device)
        posterior = Independent(Normal(mu, log_var.exp().sqrt()), 1)
        posterior_sample = posterior.rsample()
        # Compute KL divergence
        prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(log_var)), 1)
        kl_div = torch.mean(kl_divergence(posterior, prior))
        return posterior_sample, kl_div, [mu, log_var.exp().sqrt()]