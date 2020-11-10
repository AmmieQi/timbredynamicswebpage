from ddspsynth.util import slice_windows, midi_to_hz, hz_to_midi, resample_frames
from ddspsynth.spectral import compute_loudness
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddspsynth.layers import MLP
from ddspsynth.transforms import LogTransform
from torchaudio.transforms import MFCC
from ddspsynth.spectral import MelSpec, Mfcc
from ddspsynth.layers import Normalize1d, Normalize2d

def get_window_hop(enc_frame_setting):
    if enc_frame_setting not in ['coarse', 'fine', 'finer']:
        raise ValueError(
            '`enc_frame_setting` currently limited to coarse, fine, finer')
    # copied from ddsp
    # this only works when x.shape[-1] = 64000
    z_audio_spec = {
        'coarse': { # 62 or something
            'n_fft': 2048,
            'overlap': 0.5
        },
        'fine': {
            'n_fft': 1024,
            'overlap': 0.5
        },
        'finer': {
            'n_fft': 1024,
            'overlap': 0.75
        },
    }
    n_fft = z_audio_spec[enc_frame_setting]['n_fft']
    hop_length = int((1 - z_audio_spec[enc_frame_setting]['overlap']) * n_fft)
    return n_fft, hop_length

class Encoder(nn.Module):
    """
    same as the one in DDSP
    """
    def __init__(self, f0_encoder=None, encode_ld=False):
        super().__init__()
        self.f0_encoder = f0_encoder
        self.encode_ld = encode_ld

    def forward(self, conditioning):
        if self.f0_encoder:
            # Use frequency conditioning created by the f0_encoder, not the dataset.
            # Overwrite `f0_scaled` and `f0_hz`. 'f0_scaled' is a value in [0, 1]
            # corresponding to midi values [0..127]
            conditioning['f0_scaled'] = self.f0_encoder(conditioning)
            conditioning['f0_hz'] = midi_to_hz(conditioning['f0_scaled'] * 127.0)
        else:
            if len(conditioning['f0_hz'].shape) == 2:
                 # [batch, n_frames, feature_size=1]
                conditioning['f0_hz'] = conditioning['f0_hz'][:, :, None]
            conditioning['f0_scaled'] = hz_to_midi(conditioning['f0_hz']) / 127.0
        if self.encode_ld:
            conditioning = self.fill_loudness(conditioning)
            ld_time_steps = conditioning['ld_scaled'].shape[1]
        else:
            ld_time_steps = 0

        z = self.compute_z(conditioning)
        z_time_steps = z.shape[1]
        f0_time_steps = conditioning['f0_scaled'].shape[1]

        # max_time_steps = max(z_time_steps, f0_time_steps, ld_time_steps)

        # conditioning['z'] = self.expand(z, max_time_steps)
        conditioning['z'] = z
        conditioning['f0_scaled'] = self.expand(conditioning['f0_scaled'], z_time_steps)
        if self.encode_ld:
            conditioning['ld_scaled'] = self.expand(conditioning['ld_scaled'], z_time_steps)
        return conditioning

    def fill_loudness(self, conditioning):
        conditioning['ld_db'] = compute_loudness(conditioning['audio'], sample_rate=16000, frame_rate=250, n_fft=2048, range_db=120.0, ref_db=20.7).unsqueeze(-1)
        conditioning['ld_scaled'] = conditioning['ld_db'] / 120 + 1.0
        return conditioning

    def expand(self, cond, time_steps):
        """Make sure some conditioning has same temporal resolution as other conditioning."""
        # Add time dim of z if necessary.
        if len(cond.shape) == 2:
            cond = cond[:, None, :]
        # Expand time dim of cond if necessary.
        cond_time_steps = int(cond.shape[1])
        if cond_time_steps != time_steps:
            cond = resample_frames(cond, time_steps)
        return cond

    def compute_z(self, conditioning):
        """Takes in conditioning dictionary, returns a latent tensor z."""
        raise NotImplementedError

class DDSPEncoder(Encoder):
    def __init__(self, frame_setting, encoder_dims, n_mfcc=30, num_layers=1, hidden_size=512, dropout_p=0, sr=16000, f0_encoder=None, encode_ld=False):
        super().__init__(f0_encoder, encode_ld)
        n_fft, hop = get_window_hop(frame_setting)
        self.mfcc = Mfcc(n_fft, hop, 128, n_mfcc, f_min=20)
        # self.norm = Normalize1d(n_mfcc, 'instance')
        self.norm = Normalize2d('batch')
        self.gru = nn.GRU(n_mfcc, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, encoder_dims)

    def compute_z(self, conditioning):
        """
        MFCC -> Normalize -> GRU -> Dense
        original paper computes mfcc -> instance norm -> GRU -> dense
        """
        # x: [batch_size, n_frames, input_size]
        mfcc = self.mfcc(conditioning['audio']) # [batch, n_mfcc, time]
        mfcc = self.norm(mfcc).permute(0, 2, 1)
        # normalize 
        output, _hidden = self.gru(mfcc)
        return F.relu(self.out(output)) # output: [batch_size, n_frames, encoder_dims]

class MfccMlpEncoder(Encoder):
    def __init__(self, frame_setting, encoder_dims, n_mfcc=40, num_layers=3, hidden_size=256, sr=16000, f0_encoder=None, encode_ld=False):
        super().__init__(f0_encoder, encode_ld)
        n_fft, hop = get_window_hop(frame_setting)
        self.mfcc = Mfcc(n_fft, hop, 128, n_mfcc, f_min=20)
        self.norm = Normalize2d('batch')

        self.mlp = MLP(n_mfcc, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, encoder_dims)

    def compute_z(self, conditioning):
        mfcc = self.mfcc(conditioning['audio']) # [batch, n_mels, time]
        mfcc = self.norm(mfcc.permute(0, 2, 1))
        # x: [batch_size, n_frames, hidden_size]
        output = self.mlp(mfcc)
        # output: [batch_size, n_frames, latent_size]
        output = self.out(output)
        return F.relu(output)

class MelMlpEncoder(Encoder):
    def __init__(self, frame_setting, encoder_dims, n_mels=40, num_layers=3, hidden_size=256, sr=16000, f0_encoder=None, encode_ld=False):
        super().__init__(f0_encoder, encode_ld)
        n_fft, hop = get_window_hop(frame_setting)
        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop, n_mels=n_mels), LogTransform())
        self.norm = Normalize2d('batch')

        self.mlp = MLP(n_mels, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, encoder_dims)

    def compute_z(self, conditioning):
        logmel = self.logmel(conditioning['audio']) # [batch, n_mels, time]
        logmel = self.norm(logmel.permute(0, 2, 1))
        # x: [batch_size, n_frames, hidden_size]
        output = self.mlp(logmel)
        # output: [batch_size, n_frames, latent_size]
        output = self.out(output)
        return F.relu(output)

class MelConvEncoder(Encoder):
    def __init__(self, frame_setting, encoder_dims, n_mels=128, channels=64, kernel_size=7, strides=[2,2,2,2], hidden_size=256, sr=16000, f0_encoder=None, encode_ld=False):
        super().__init__(f0_encoder, encode_ld)
        n_fft, hop = get_window_hop(frame_setting)
        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop, n_mels=n_mels), LogTransform())
        self.frame_size = n_mels
        self.norm = Normalize2d('batch')
        self.channels = channels
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, channels, kernel_size,
                        padding=kernel_size // 2,
                        stride=strides[0]), nn.BatchNorm1d(channels), nn.ReLU())]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[i]), nn.BatchNorm1d(channels), nn.ReLU())
                         for i in range(1, len(strides) - 1)]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[-1]))])
        self.l_out = self.get_downsampled_length()[-1]
        self.mlp = MLP(self.l_out * channels, encoder_dims, loop=2)

    def compute_z(self, conditioning):
        x = self.logmel(conditioning['audio']) # [batch, n_mels, time]
        batch_size, n_mels, n_frames = x.shape
        x = self.norm(x.permute(0, 2, 1))
        x = x.view(-1, n_mels).unsqueeze(1)
        # x: [batch_size*n_frames, 1, n_mels]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = torch.relu(x)
        x = x.view(batch_size, n_frames, self.channels, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        output = self.mlp(x)
        # output: [batch_size, n_frames, latent_size]
        return output
    
    def get_downsampled_length(self):
        l = self.frame_size
        lengths = [l]
        for conv in self.convs:
            conv_module = conv[0]
            l = (l + 2 * conv_module.padding[0] - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1) - 1) // conv_module.stride[0] + 1
            lengths.append(l)
        return lengths

class WaveEncoder(Encoder):
    """
    Split the input waveform and downsample it with 1DConvNets
    """
    def __init__(self, frame_setting, encoder_dims, channels, kernel_size, strides, f0_encoder=None, encode_ld=False):
        super().__init__(f0_encoder, encode_ld)
        n_fft, hop = get_window_hop(frame_setting)
        self.frame_size = n_fft # same as window size
        self.hop_size = hop

        self.encoder_dims = encoder_dims
        self.convs = nn.ModuleList(
                [nn.Sequential(nn.Conv1d(1, channels, kernel_size,
                            padding=kernel_size // 2,
                            stride=strides[0]), nn.ReLU())]
                + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                            padding=kernel_size // 2,
                            stride=strides[i]), nn.ReLU()) 
                            for i in range(1, len(strides) - 1)]
                + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                            padding=kernel_size // 2,
                            stride=strides[-1]))])
        self.l_out = self.get_downsampled_length()[-1]
        self.mlp = MLP(self.l_out * encoder_dims, encoder_dims, loop=2)

    def get_downsampled_length(self):
        l = self.frame_size
        lengths = [l]
        for conv in self.convs:
            l = (l + 2*conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
            lengths.append(l)
        return lengths

    def compute_z(self, conditioning):
        x = conditioning['audio']
        batch_size, l_x = x.shape
        x = slice_windows(x, self.frame_size, self.hop_size) #batch, frame_size, n_frames
        n_frames = x.shape[2]
        x = x.permute(0, 2, 1).reshape(-1, self.frame_size) #batch*n_frames, frame_size
        x = x[:, None, :]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.relu(x)
        x = x.view(batch_size, n_frames, self.encoder_dims, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        return self.mlp(x)
