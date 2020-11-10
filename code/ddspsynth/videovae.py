import torch
import torch.nn as nn
from ddspsynth.aes import AE
from ddspsynth.encoders import get_window_hop
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence
from ddspsynth.util import resample_frames, hz_to_midi
import math

class Psi(nn.Module):
    def __init__(self, in_size=64, out_size=64, hidden_size=16):
        super().__init__()
        self.linear_mu = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.modules.normalization.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size))
        self.linear_logscale = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.modules.normalization.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size))

    def forward(self, mu, logscale):
        mu = self.linear_mu(mu)
        logscale = self.linear_logscale(logscale)
        return mu, logscale

class Psi2(nn.Module):
    def __init__(self, in_size=64, out_size=64, hidden_size=16):
        super().__init__()
        self.linear_mu = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size))
        self.linear_logscale = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size))

    def forward(self, mu, logscale):
        """
        [batch, latent_dim] or [batch, n_frames, latent_dims]
        """
        orig_shape = mu.shape
        if len(orig_shape) > 2:
            mu = mu.flatten(0,1)
            logscale = logscale.flatten(0,1)
        mu = self.linear_mu(mu)
        logscale = self.linear_logscale(logscale)
        if len(orig_shape) > 2:
            mu = mu.view(orig_shape[0], orig_shape[1], -1)
            logscale = logscale.view(orig_shape[0], orig_shape[1], -1)
        return mu, logscale

class RNNPriorVAE(AE):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__(encoder, decoder, encoder_dims, latent_dims)
        # Structured Latent space 
        self.psi_q = Psi(encoder_dims, latent_dims)
        self.temporal_q = nn.GRU(latent_dims*2, latent_dims, batch_first=True)
        self.psi_dy = Psi(latent_dims*2, latent_dims)
        self.psi_p = Psi(latent_dims, latent_dims)
        self.apply(self.init_parameters)

    def temporal_latent_model(self, mu_q, logscale_q, h_0):
        psi = torch.cat([mu_q, logscale_q], dim=-1)
        temp_q, _hidden = self.temporal_q(psi, h_0) # one off
        temp_q = torch.cat([h_0.permute(1, 0, 2), temp_q[:, :-1, :]], dim=1) #0~T-1
        return temp_q

    def mix_with_temp(self, mu, logscale, temp_q):
        # mix with temporal info
        mu = torch.cat([mu, temp_q], dim=-1)
        logscale = torch.cat([logscale, temp_q], dim=-1)
        mu, logscale = self.psi_dy(mu, logscale)
        return mu, logscale.exp()

    def latent(self, conditioning):
        """
        z_enc: [batch, frames, latent_dims]
        """
        z_enc = conditioning['z']
        batch_size, n_frames, _encdims = z_enc.shape
        mu_q, logscale_q = self.psi_q(z_enc, z_enc)
        # feed into temporal model
        h_0 = torch.randn(1, batch_size, self.latent_dims).to(z_enc.device)*0.01
        temp_q = self.temporal_latent_model(mu_q, logscale_q, h_0)
        # final posterior distribution with rnn information
        mu_z, scale_z = self.mix_with_temp(mu_q, logscale_q, temp_q)
        posterior = Independent(Normal(mu_z, scale_z), 1)
        posterior_sample = posterior.rsample()
        # prior distribution with rnn information
        mu, scale = self.psi_p(temp_q, temp_q)
        scale = scale.exp()
        prior = Independent(Normal(mu, scale), 1)
        # prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(scale)), 1)
        # sum over latent dim mean over batch
        kl_div = torch.mean(kl_divergence(posterior, prior))
        return posterior_sample, kl_div, [mu_z, scale_z]
    
    def generate(self, synth, h_0, f0_hz, enc_frame_setting='fine', n_samples=16000):
        """
        synth:          synth to generate audio
        h_0:            initial seed of RNN [batch, latent_dims]
        f0_hz:          f0 conditioning of synth [batch, f0_n_frames, 1]
        enc_frame_setting: fft/hop size
        n_samples:      output audio length in samples
        """
        if len(h_0.shape) == 2:
            h = h_0[None, :, :] # 1, batch, latent_dims 
        else:
            h = h_0
        n_fft, hop_length = get_window_hop(enc_frame_setting)
        n_frames = math.ceil((n_samples - n_fft) / hop_length) + 1
        f0_hz = resample_frames(f0_hz, n_frames) # needs to have same dimension as z
        params_list = []
        for i in range(n_frames):
            cond = {}
            mu, logscale = self.psi_p(h.permute(1, 0, 2), h.permute(1, 0, 2))
            scale = logscale.exp()
            prior = Independent(Normal(mu, scale), 1)
            prior_sample = prior.rsample()
            cond['z'] = prior_sample
            cond['f0_hz'] = f0_hz[:, i, :].unsqueeze(1)
            cond['f0_scaled'] = hz_to_midi(cond['f0_hz']) / 127.0
            # generate x
            y = self.decode(cond)
            params = synth.fill_params(y, cond)
            params_list.append(params)
            x_tilde, _outputs = synth(params, n_samples=n_fft) # write exactly one frame
            cond['audio'] = x_tilde
            # encode
            cond = self.encoder(cond)
            z_enc = cond['z']
            # get psi_q
            mu, logscale = self.psi_q(z_enc, z_enc)
            psi = torch.cat([mu, logscale], dim=-1)
            # temporal model
            temp_q, h = self.temporal_q(psi, h) # one off

        param_names = params_list[0].keys()
        final_params = {}
        for pn in param_names:
            #cat over frames
            final_params[pn] = torch.cat([par[pn] for par in params_list], dim=1)
        
        final_audio, _outputs = synth(final_params, n_samples=n_samples)
        return final_params, final_audio

class VideoVAE(RNNPriorVAE):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, attribute_dims):
        super().__init__(encoder, decoder, encoder_dims, latent_dims)
        # Mix with attributes 
        self.psi_a = Psi(latent_dims + attribute_dims, latent_dims)
        self.psi_p = Psi(latent_dims + attribute_dims, latent_dims)
        self.apply(self.init_parameters)
    
    def attribute_latent(self, mu_q, logscale_q, attributes):
        """
        mix psi_q with attribute to get psi_a
        """
        # mix with attributes maybe FiLM would be better
        mu = torch.cat([mu_q, attributes], dim=-1)
        logscale = torch.cat([logscale_q, attributes], dim=-1)
        mu, logscale = self.psi_a(mu, logscale)
        return mu, logscale

    def latent(self, conditioning):
        """
        z_enc: [batch, frames, latent_dims]
        attributes: [batch, frames, attribute_dims]
        """
        z_enc = conditioning['z']
        attributes = conditioning['attributes']
        batch_size, n_frames, _encdims = z_enc.shape
        if len(attributes.shape) < 3:
            # expand along frame dimension
            attributes = attributes.unsqueeze(1).expand(-1, n_frames, -1)
        mu_q, logscale_q = self.psi_q(z_enc, z_enc)
        # mix with latent
        mu_a, logscale_a = self.attribute_latent(mu_q, logscale_q, attributes)
        # feed into temporal model
        h_0 = torch.rand(1, batch_size, self.latent_dims).to(z_enc.device)*0.01
        temp_q = self.temporal_latent_model(mu_q, logscale_q, h_0)
        # posterior distribution
        mu_z, scale_z = self.mix_with_temp(mu_a, logscale_a, temp_q)
        posterior = Independent(Normal(mu_z, scale_z), 1)
        posterior_sample = posterior.rsample()
        # prior
        output = torch.cat([temp_q, attributes], dim=-1)
        mu, scale = self.psi_p(output, output)
        scale = scale.exp()
        prior = Independent(Normal(mu, scale), 1)
        # sum over latent dim mean over batch
        kl_div = torch.mean(kl_divergence(posterior, prior))
        return posterior_sample, kl_div, [mu_z, scale_z]
    
    def generate(self, synth, h_0, f0_hz, attributes, enc_frame_setting='fine', n_samples=16000):
        """
        synth:          synth to generate audio
        h_0:            initial seed of RNN [batch, latent_dims]
        f0_hz:          f0 conditioning of synth [batch, f0_n_frames, 1]
        attributes:     attributes [batch, n_frames, attribute_size]
        enc_frame_setting: fft/hop size
        n_samples:      output audio length in samples
        """
        if len(h_0.shape) == 2:
            h = h_0[None, :, :] # 1, batch, latent_dims 
        else:
            h = h_0
        n_fft, hop_length = get_window_hop(enc_frame_setting)
        n_frames = math.ceil((n_samples - n_fft) / hop_length) + 1
        f0_hz = resample_frames(f0_hz, n_frames) # needs to have same dimension as z
        params_list = []
        for i in range(n_frames):
            cond = {}
            output = torch.cat([h.permute(1, 0, 2), attributes], dim=-1)
            mu, logscale = self.psi_p(output, output)
            scale = logscale.exp()
            prior = Independent(Normal(mu, scale), 1)
            prior_sample = prior.rsample()
            cond['z'] = prior_sample
            cond['f0_hz'] = f0_hz[:, i, :].unsqueeze(1)
            cond['f0_scaled'] = hz_to_midi(cond['f0_hz']) / 127.0
            # generate x
            y = self.decode(cond)
            params = synth.fill_params(y, cond)
            params_list.append(params)
            x_tilde, _outputs = synth(params, n_samples=n_fft) # write exactly one frame
            cond['audio'] = x_tilde
            # encode
            cond = self.encoder(cond)
            z_enc = cond['z']
            # get psi_q
            mu, logscale = self.psi_q(z_enc, z_enc)
            psi = torch.cat([mu, logscale], dim=-1)
            # temporal model
            temp_q, h = self.temporal_q(psi, h) # one off

        param_names = params_list[0].keys()
        final_params = {}
        for pn in param_names:
            #cat over frames
            final_params[pn] = torch.cat([par[pn] for par in params_list], dim=1)
        
        final_audio, _outputs = synth(final_params, n_samples=n_samples)
        return final_params, final_audio

class CondRNNPriorVAE(AE):
    # Conditional Sampling for videovae
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__(encoder, decoder, encoder_dims, latent_dims)
        # Structured Latent space 
        self.psi_q = Psi2(encoder_dims, latent_dims)
        # takes z as input
        self.temporal_q = nn.GRUCell(latent_dims*3, latent_dims)
        self.psi_dy = Psi2(latent_dims*2, latent_dims)
        self.psi_p = Psi2(latent_dims, latent_dims)
        self.h_process = Psi2(latent_dims, latent_dims)
        self.apply(self.init_parameters)        

    def temporal_model_step(self, mu_q_t, logscale_q_t, h, attribute=None):
        """
        generate z_t autoregressively
        """
        # mix with temporal info
        h_mu, h_scale = self.h_process(h, h)
        mu = torch.cat([mu_q_t, h_mu], dim=-1)
        logscale = torch.cat([logscale_q_t, h_scale], dim=-1)
        mu_z_t, logscale_z_t = self.psi_dy(mu, logscale)
        scale_z_t = logscale_z_t.exp()
        # final posterior distribution with rnn information
        posterior_t = Independent(Normal(mu_z_t, scale_z_t), 1)
        posterior_sample_t = posterior_t.rsample()
        # prior distribution with rnn information
        if not attribute is None:
            mixed_h_mu = torch.cat([h_mu, attribute], dim=-1)
            mixed_h_scale = torch.cat([h_scale, attribute], dim=-1)
            mu_p_t, logscale_p_t = self.psi_p(mixed_h_mu, mixed_h_scale)
        else:
            mu_p_t, logscale_p_t = self.psi_p(h_mu, h_scale)
        # scale_p_t = logscale_p_t.exp()
        scale_p_t = torch.ones_like(logscale_p_t)
        prior_t = Independent(Normal(mu_p_t, scale_p_t), 1)
        kl_div_t = torch.mean(kl_divergence(posterior_t, prior_t))
        return posterior_sample_t, kl_div_t, mu_z_t, scale_z_t
        
    def latent(self, conditioning):
        """
        z_enc: [batch, frames, latent_dims]
        """
        z_enc = conditioning['z']
        batch_size, n_frames, _encdims = z_enc.shape
        mu_q, logscale_q = self.psi_q(z_enc, z_enc)
        h = torch.randn(batch_size, self.latent_dims).to(z_enc.device)*0.01
        kl_div = 0
        mu_z = torch.zeros_like(mu_q)
        scale_z = torch.zeros_like(logscale_q)
        z = torch.zeros_like(mu_q)
        for t in range(n_frames):
            z_t, kl_t, mu_z_t, scale_z_t = self.temporal_model_step(mu_q[:,t,:], logscale_q[:,t,:], h)
            kl_div += kl_t
            z[:,t,:] = z_t
            mu_z[:,t,:] = mu_z_t
            scale_z[:,t,:] = scale_z_t
            rnn_input = torch.cat([mu_q[:, t, :], logscale_q[:, t, :], z_t], dim=-1)
            # feed z_t into RNN 
            h = self.temporal_q(rnn_input, h)
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
            h_mu, h_scale = self.h_process(h, h)
            mu_t, logscale_t = self.psi_p(h_mu, h_scale) # [batch, latent_size]
            scale_t = logscale_t.exp()
            prior_t = Independent(Normal(mu_t, scale_t), 1)
            prior_sample_t = prior_t.rsample()
            cond = {}
            z[:,t,:] = prior_sample_t
            cond['z'] = prior_sample_t.unsqueeze(1)
            cond['f0_hz'] = f0_hz[:, t, :].unsqueeze(1)
            cond['f0_scaled'] = hz_to_midi(cond['f0_hz']) / 127.0
            # generate x
            y = self.decode(cond)
            params = synth.fill_params(y, cond)
            params_list.append(params)
            x_tilde, _outputs = synth(params, n_samples=n_fft) # write exactly one frame
            cond['audio'] = x_tilde
            # encode
            cond = self.encoder(cond)
            z_enc = cond['z'].squeeze(1)
            # get psi_q
            mu, logscale = self.psi_q(z_enc, z_enc)
            rnn_input = torch.cat([mu, logscale, prior_sample_t], dim=-1)
            # temporal model
            h = self.temporal_q(rnn_input, h) # one off

        cond = {}
        cond['z'] = z
        cond['f0_hz'] = f0_hz
        y_params = self.decode(cond)
        params = synth.fill_params(y_params, cond)
        resyn_audio, outputs = synth(params, n_samples)
        return params, resyn_audio

class CondVideoVAE(CondRNNPriorVAE):
    # Conditional Sampling for videovae
    def __init__(self, encoder, decoder, encoder_dims, latent_dims, attribute_dims):
        super().__init__(encoder, decoder, encoder_dims, latent_dims)
        # Mix with attributes 
        self.psi_a = Psi2(latent_dims + attribute_dims, latent_dims)
        self.psi_p = Psi2(latent_dims + attribute_dims, latent_dims)
        self.apply(self.init_parameters)

    def attribute_latent(self, mu_q, logscale_q, attributes):
        """
        mix psi_q with attribute to get psi_a
        """
        # mix with attributes maybe FiLM would be better
        mu = torch.cat([mu_q, attributes], dim=-1)
        logscale = torch.cat([logscale_q, attributes], dim=-1)
        mu, logscale = self.psi_a(mu, logscale)
        return mu, logscale

    def latent(self, conditioning):
        """
        z_enc: [batch, frames, latent_dims]
        """
        z_enc = conditioning['z']
        attributes = conditioning['attributes']
        batch_size, n_frames, _encdims = z_enc.shape
        if len(attributes.shape) < 3:
            # expand along frame dimension
            attributes = attributes.unsqueeze(1).expand(-1, n_frames, -1)
        mu_q, logscale_q = self.psi_q(z_enc, z_enc)
        # mix with latent
        mu_a, logscale_a = self.attribute_latent(mu_q, logscale_q, attributes)
        h = torch.randn(batch_size, self.latent_dims).to(z_enc.device)*0.01
        kl_div  = 0
        mu_z = torch.zeros_like(mu_q)
        scale_z = torch.zeros_like(logscale_q)
        z = torch.zeros_like(mu_q)
        for t in range(n_frames):
            z_t, kl_t, mu_z_t, scale_z_t = self.temporal_model_step(mu_a[:,t,:], logscale_a[:,t,:], h, attributes[:, t, :])
            kl_div += kl_t
            z[:,t,:] = z_t
            mu_z[:,t,:] = mu_z_t
            scale_z[:,t,:] = scale_z_t
            # feed z_t into RNN 
            rnn_input = torch.cat([mu_q[:, t, :], logscale_q[:, t, :], z_t], dim=-1)
            h = self.temporal_q(rnn_input, h)
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
        h = h_0 # initial state
        n_fft, hop_length = get_window_hop(enc_frame_setting)
        n_frames = math.ceil((n_samples - n_fft) / hop_length) + 1
        f0_hz = resample_frames(f0_hz, n_frames) # needs to have same dimension as z
        params_list = []
        z = torch.zeros(h_0.shape[0], n_frames, self.latent_dims).to(h.device)
        if len(attributes.shape) == 2:
            attributes = attributes[:, None, :].expand(-1, n_frames, -1)
        for t in range(n_frames):
            cond = {}
            # mix with attribute
            mixed_h = torch.cat([h, attributes[:, t, :]], dim=-1)
            mu_t, logscale_t = self.psi_p(mixed_h, mixed_h) # [batch, latent_size]
            scale_t = logscale_t.exp()
            prior_t = Independent(Normal(mu_t, scale_t), 1)
            prior_sample_t = prior_t.rsample()
            cond = {}
            cond['z'] = prior_sample_t.unsqueeze(1)
            z[:,t,:] = prior_sample_t
            cond['f0_hz'] = f0_hz[:, t, :].unsqueeze(1)
            cond['f0_scaled'] = hz_to_midi(cond['f0_hz']) / 127.0
            # generate x
            y = self.decode(cond)
            params = synth.fill_params(y, cond)
            params_list.append(params)
            x_tilde, _outputs = synth(params, n_samples=n_fft) # write exactly one frame
            cond['audio'] = x_tilde
            # encode
            cond = self.encoder(cond)
            z_enc = cond['z'].squeeze(1)
            # get psi_q
            mu, logscale = self.psi_q(z_enc, z_enc)
            rnn_input = torch.cat([mu, logscale, prior_sample_t], dim=-1)
            # temporal model
            h = self.temporal_q(rnn_input, h) # one off

        cond = {}
        cond['z'] = z
        cond['f0_hz'] = f0_hz
        y_params = self.decode(cond)
        params = synth.fill_params(y_params, cond)
        resyn_audio, outputs = synth(params, n_samples)
        return params, resyn_audio