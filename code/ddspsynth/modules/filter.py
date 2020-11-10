import torch
import torch.nn as nn
from ddspsynth.synth import Processor
import ddspsynth.util as util
import numpy as np
import math

class FIRFilter(Processor):
    """
    taken from ddsp-pytorch and ddsp
    uses frequency sampling
    """
    
    def __init__(self, filter_size=64, scale_fn=util.exp_sigmoid, name='firfilter', initial_bias=-5.0):
        super().__init__()
        self.filter_size = filter_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def forward(self, audio, freq_response):
        """pass audio through FIRfilter
        Args:
            audio (torch.Tensor): [batch, n_samples]
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        if self.scale_fn is not None:
            freq_response = self.scale_fn(freq_response + self.initial_bias)
        return util.fir_filter(audio, freq_response, self.filter_size)
    
    def get_param_sizes(self):
        return {'freq_response': self.filter_size // 2 + 1, 'audio': None}

class IIRLPF(Processor):
    """biquad lowpass -12db/oct
    """
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        # window is the same size as the number of frequency bins

    def forward(self, audio, cutoff_freq, Q):
        """pass audio through FIRfilter
        Args:
            audio (torch.Tensor): [batch, n_samples]
            cutoff_frequency (torch.Tensor): cutoff frequency
            Q (torch.Tensor): q factor 0~1
        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        Q = torch.clamp(Q, 0.01, 1-0.01)
        return lowpass_biquad(audio, self.sample_rate, cutoff_freq, Q)
    
    def get_param_sizes(self):
        return {'cutoff_freq': 1, 'Q': 1}

# Just copied this from torchaudio it should be differentiable 
def lowpass_biquad(waveform, sample_rate, cutoff_freq, Q):
    r"""Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)

def biquad(
        waveform: torch.Tensor,
        b0: float,
        b1: float,
        b2: float,
        a0: float,
        a1: float,
        a2: float
) -> torch.Tensor:
    r"""Perform a biquad filter of input tensor.  Initial conditions set to 0.
    https://en.wikipedia.org/wiki/Digital_biquad_filter

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        b0 (float): numerator coefficient of current input, x[n]
        b1 (float): numerator coefficient of input one time step ago x[n-1]
        b2 (float): numerator coefficient of input two time steps ago x[n-2]
        a0 (float): denominator coefficient of current output y[n], typically 1
        a1 (float): denominator coefficient of current output y[n-1]
        a2 (float): denominator coefficient of current output y[n-2]

    Returns:
        Tensor: Waveform with dimension of `(..., time)`
    """

    device = waveform.device
    dtype = waveform.dtype

    output_waveform = lfilter(
        waveform,
        torch.Tensor([a0, a1, a2], dtype=dtype, device=device),
        torch.Tensor([b0, b1, b2], dtype=dtype, device=device)
    )
    return output_waveform

def lfilter(
        waveform: torch.Tensor,
        a_coeffs: torch.Tensor,
        b_coeffs: torch.Tensor,
        clamp: bool = True,):
    r"""Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (Tensor): audio waveform of dimension of ``(..., time)``.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of ``(n_order + 1)``.
                                Lower delays coefficients are first, e.g. ``[a0, a1, a2, ...]``.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of ``(n_order + 1)``.
                                 Lower delays coefficients are first, e.g. ``[b0, b1, b2, ...]``.
                                 Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If ``True``, clamp the output signal to be in the range [-1, 1] (Default: ``True``)

    Returns:
        Tensor: Waveform with dimension of ``(..., time)``.
    """
    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 2)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_waveform, window_idxs))

    input_signal_windows.div_(a_coeffs[0])
    a_coeffs_flipped.div_(a_coeffs[0])
    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)]
        o0.addmv_(windowed_output_signal, a_coeffs_flipped, alpha=-1)
        padded_output_waveform[:, i_sample + n_order - 1] = o0

    output = padded_output_waveform[:, (n_order - 1):]

    if clamp:
        output = torch.clamp(output, min=-1., max=1.)

    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])

    return output