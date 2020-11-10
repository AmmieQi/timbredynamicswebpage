import torch
import torch.nn as nn
from ddspsynth.aes import AE
from ddspsynth.encoders import get_window_hop
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence
from ddspsynth.util import resample_frames, hz_to_midi
import math
from ddspsynth.layers import MLP, FiLM, FiLMMLP

class RNNLatent(AE):
    # basically VRNN but RNN is fed only past latent vars not input data 
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, hidden_size=64):
        super().__init__(encoder, decoder, encoder_dims, latent_dims)
        # RNN that takes past latents and attribute as condition
        self.hidden_size = hidden_size
        self.temporal = nn.GRUCell(latent_dims, hidden_size)
        self.mix_lin = MLP(hidden_size+encoder_dims, latent_dims)
        self.post_loc_out = nn.Linear(latent_dims, latent_dims)
        self.post_logscale_out = nn.Linear(latent_dims, latent_dims)
        self.prior_lin = MLP(hidden_size, latent_dims)
        self.prior_loc_out = nn.Linear(latent_dims, latent_dims)
        self.prior_logscale_out = nn.Linear(latent_dims, latent_dims)

    def get_prior(self, h):
        z_prior = self.prior_lin(h)
        mu_p_t = self.prior_loc_out(z_prior)
        scale_p_t = self.prior_logscale_out(z_prior).exp()
        return mu_p_t, scale_p_t

    def get_posterior(self, h, z_enc_t):
        z_mixed = self.mix_lin(torch.cat([h, z_enc_t], dim=-1))
        mu_z_t = self.post_loc_out(z_mixed)
        scale_z_t = self.post_logscale_out(z_mixed).exp()
        return mu_z_t, scale_z_t

    def temporal_model_step(self, z_enc_t, h, attribute=None):
        """
        generate z_t autoregressively
        """
        # mix with temporal info
        mu_z_t, scale_z_t = self.get_posterior(h, z_enc_t)
        scale_z_t = scale_z_t + 1e-4 # minimum
        # final posterior distribution with rnn information
        posterior_t = Independent(Normal(mu_z_t, scale_z_t), 1)
        posterior_sample_t = posterior_t.rsample()
        # prior distribution with rnn information
        mu_p_t, scale_p_t = self.get_prior(h)
        scale_p_t = scale_p_t + 1e-4 # minimum
        prior_t = Independent(Normal(mu_p_t, scale_p_t), 1)
        kl_div_t = torch.mean(kl_divergence(posterior_t, prior_t))
        return posterior_sample_t, kl_div_t, mu_z_t, scale_z_t
        
    def latent(self, conditioning):
        """
        z_enc: [batch, frames, latent_dims]
        """
        z_enc = conditioning['z']
        batch_size, n_frames, _encdims = z_enc.shape
        h = torch.zeros(batch_size, self.hidden_size).to(z_enc.device)
        kl_div = 0
        z = torch.zeros(batch_size, n_frames, self.latent_dims).to(z_enc.device)
        mu_z = torch.zeros_like(z)
        scale_z = torch.zeros_like(z)
        for t in range(n_frames):
            z_t, kl_t, mu_z_t, scale_z_t = self.temporal_model_step(z_enc[:, t, :], h)
            kl_div += kl_t
            z[:,t,:] = z_t
            mu_z[:,t,:] = mu_z_t
            scale_z[:,t,:] = scale_z_t
            # feed z_t into RNN
            h = self.temporal(z_t, h)
        # sum over latent dim mean over batch
        kl_div /= n_frames
        return z, kl_div, [mu_z, scale_z]
    
    def generate(self, synth, h_0, f0_hz, enc_frame_setting='fine', n_samples=16000):
        """
        synth:          synth to generate audio
        h_0:            initial state of RNN [batch, latent_dims]
        f0_hz:          f0 conditioning of synth [batch, f0_n_frames, 1]
        enc_frame_setting: fft/hop size
        n_samples:      output audio length in samples
        """
        h = h_0
        n_fft, hop_length = get_window_hop(enc_frame_setting)
        n_frames = math.ceil((n_samples - n_fft) / hop_length) + 1
        f0_hz = resample_frames(f0_hz, n_frames) # needs to have same dimension as z
        params_list = []
        z = torch.zeros(h_0.shape[0], n_frames, self.latent_dims).to(h.device)
        for t in range(n_frames):
            # prior distribution with rnn information
            mu_p_t, scale_p_t = self.get_prior(h)    
            prior_t = Independent(Normal(mu_p_t, scale_p_t), 1)
            prior_sample_t = prior_t.rsample()
            h = self.temporal(prior_sample_t, h)
            z[:,t,:] = prior_sample_t
        cond = {}
        cond['z'] = z
        cond['f0_hz'] = f0_hz
        y_params = self.decode(cond)
        params = synth.fill_params(y_params, cond)
        resyn_audio, outputs = synth(params, n_samples)
        return params, resyn_audio
        
class AttrRNNLatent(RNNLatent):
    # basically VRNN but RNN is fed only past latent vars not input data 
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, attribute_dims, hidden_size=64):
        super().__init__(encoder, decoder, encoder_dims, latent_dims, hidden_size)
        # modulate h with attributes using film
        # self.film_attr = FiLM(hidden_size, attribute_dims)
        self.temporal = nn.GRUCell(latent_dims+attribute_dims, hidden_size)

    def latent(self, conditioning):
        """
        z_enc: [batch, frames, latent_dims]
        """
        attributes = conditioning['attributes']
        z_enc = conditioning['z']
        batch_size, n_frames, _encdims = z_enc.shape
        h = torch.zeros(batch_size, self.hidden_size).to(z_enc.device)
        kl_div = 0
        z = torch.zeros(batch_size, n_frames, self.latent_dims).to(z_enc.device)
        mu_z = torch.zeros_like(z)
        scale_z = torch.zeros_like(z)
        if len(attributes.shape) == 2:
            attributes = attributes[:, None, :].expand(-1, n_frames, -1)
        # set up initial prior with attributes
        z_t = torch.zeros(batch_size, self.latent_dims).to(z_enc.device)
        rnn_input = torch.cat([z_t, attributes[:, 0, :]], dim=-1)
        h = self.temporal(rnn_input, h)
        for t in range(n_frames):
            z_t, kl_t, mu_z_t, scale_z_t = self.temporal_model_step(z_enc[:, t, :], h)
            kl_div += kl_t
            z[:,t,:] = z_t
            mu_z[:,t,:] = mu_z_t
            scale_z[:,t,:] = scale_z_t
            rnn_input = torch.cat([z_t, attributes[:, t, :]], dim=-1) # condition LSTM on a
            # feed z_t into RNN
            h = self.temporal(rnn_input, h)
        # sum over latent dim mean over batch
        kl_div /= n_frames
        return z, kl_div, [mu_z, scale_z]
    
    def generate(self, synth, h_0, f0_hz, attributes, enc_frame_setting='fine', n_samples=16000):
        """
        synth:          synth to generate audio
        h_0:            initial state of RNN [batch, latent_dims]
        f0_hz:          f0 conditioning of synth [batch, f0_n_frames, 1]
        attributes:     attributes [batch, attribute_size] or [batch, n_frames, attribute_size]
        enc_frame_setting: fft/hop size
        n_samples:      output audio length in samples
        """
        n_fft, hop_length = get_window_hop(enc_frame_setting)
        n_frames = math.ceil((n_samples - n_fft) / hop_length) + 1
        f0_hz = resample_frames(f0_hz, n_frames).to(h_0.device) # needs to have same dimension as z
        params_list = []
        z = torch.zeros(h_0.shape[0], n_frames, self.latent_dims).to(h_0.device)
        if len(attributes.shape) == 2:
            attributes = attributes[:, None, :].expand(-1, n_frames, -1)
        # set up initial prior with attributes
        z_t = torch.zeros(h_0.shape[0], self.latent_dims).to(h_0.device)
        rnn_input = torch.cat([z_t, attributes[:, 0, :]], dim=-1)
        h = self.temporal(rnn_input, h_0)
        for t in range(n_frames):
            # prior distribution with rnn information
            mu_p_t, scale_p_t = self.get_prior(h)    
            prior_t = Independent(Normal(mu_p_t, scale_p_t), 1)
            z_t = prior_t.rsample()
            rnn_input = torch.cat([z_t, attributes[:, t, :]], dim=-1)
            h = self.temporal(rnn_input, h)
            z[:,t,:] = z_t
        cond = {}
        cond['z'] = z
        cond['f0_hz'] = f0_hz
        y_params = self.decode(cond)
        params = synth.fill_params(y_params, cond)
        resyn_audio, outputs = synth(params, n_samples)
        return params, resyn_audio

class FiLMRNNLatent(RNNLatent):
    # Use FiLM to mix h with z
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, hidden_size=64):
        super().__init__(encoder, decoder, encoder_dims, latent_dims, hidden_size)
        # RNN that takes past latents and attribute as condition
        self.mix_lin = FiLMMLP(encoder_dims, latent_dims, hidden_size)

    def get_posterior(self, h, z_enc_t):
        z_mixed = self.mix_lin(z_enc_t, h)
        mu_z_t = self.post_loc_out(z_mixed)
        scale_z_t = self.post_logscale_out(z_mixed).exp()
        return mu_z_t, scale_z_t

class FiLMAttrRNNLatent(AttrRNNLatent):
    # Use FiLM to mix h with z
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, attribute_dims, hidden_size=64):
        super().__init__(encoder, decoder, encoder_dims, latent_dims, attribute_dims, hidden_size)
        # RNN that takes past latents and attribute as condition
        self.mix_lin = FiLMMLP(encoder_dims, latent_dims, hidden_size)

    def get_posterior(self, h, z_enc_t):
        z_mixed = self.mix_lin(z_enc_t, h)
        mu_z_t = self.post_loc_out(z_mixed)
        scale_z_t = self.post_logscale_out(z_mixed).exp()
        return mu_z_t, scale_z_t