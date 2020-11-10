import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ddspsynth.transforms import LogTransform
from ddspsynth.spectral import MelSpec
from ddspsynth.layers import MLP

class Classifier(nn.Module):
    """
    For calculating inception score
    """
    def __init__(self, n_classes, n_fft=1024, hop_length=512, n_mels=128, channels=64, length=4.0, sample_rate=16000):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(1, channels, kernel_size=5, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(channels, channels, kernel_size=5, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(channels, channels, kernel_size=5, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(channels, channels, kernel_size=5, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_samples = length * sample_rate
        self.n_frames = math.ceil((n_samples - n_fft) / hop_length) + 1
        self.logmel = nn.Sequential(MelSpec(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels), LogTransform())
        # get_final_size
        dummy = torch.randn(1, 1, n_mels, self.n_frames)
        dummy = self.convs(dummy)
        self.conv_shape = list(dummy.shape[2:])
        self.mlp = MLP(channels*self.conv_shape[0]*self.conv_shape[1], 64, loop=2)
        self.out = nn.Linear(64, n_classes)

    def forward(self, audio):
        """
        audio: raw waveform (batch, n_samples)
        label: one-hot (batch, n_classes)
        """
        logmel = self.logmel(audio)
        output = self.convs(logmel.unsqueeze(1))
        output = output.flatten(1,3)
        output = self.mlp(output)
        return self.out(output)

    def intra_entropy(self, data_dict, device):
        """
        Confidence in classification
        H[p(y|x)]
        """
        data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
        out = self(data_dict['audio'])
        out = F.softmax(out, dim=-1)
        entropy = (out*out.log()).sum(dim=-1)

    def train_epoch(self, loader, loss, optimizer, device):
        self.train()
        total_loss = 0
        for data_dict in loader:
            data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
            out = self(data_dict['audio'])
            _, label = torch.max(data_dict['instrument'], 1)
            # Reconstruction loss
            batch_loss = loss(out, label)
            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.detach().cpu()
        total_loss /= len(loader)
        return total_loss

    def eval_epoch(self, loader, device):
        self.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in data_dict.items()}
                out = self(data_dict['audio'])
                _, label = torch.max(data_dict['instrument'], 1)
                _, predicted = torch.max(out.data, 1)
                total += label.shape[0] # batch size
                correct += (predicted == label).sum().detach().cpu()
        accuracy = correct/float(total)
        return accuracy