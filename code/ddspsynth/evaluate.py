import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import tqdm
from scipy import signal
import librosa
from librosa.feature import melspectrogram
from sklearn import decomposition
import os, json, argparse, pickle, csv, math
from ddspsynth.attnsynth import AttnSynth, AttnDecoder, RnnEncoder
from ddspsynth.data import load_dataset
from ddspsynth.spectral import SpectralLoss, spectrogram
from ddspsynth.transforms import LogTransform, Permute
from ddspsynth.modules.generators import Additive
from ddspsynth.synth import Synthesizer
from ddspsynth.plot import plot_recons
from ddspsynth.videovae import VideoVAE, RNNPriorVAE, CondRNNPriorVAE, CondVideoVAE
from ddspsynth.dynamicaes import AttrRNNLatent, RNNLatent, FiLMAttrRNNLatent, FiLMRNNLatent
from ddspsynth.encoders import get_window_hop

# def eval_model(model, test_loader, recon_loss, base_dir, input_key='mel', loss_type='L1'):
#     self.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for data_dict in test_loader:
#             target = data_dict['audio'].to(device, non_blocking=True)
#             input_data = data_dict[input_key].to(device, non_blocking=True)
#             # f0 = data_dict['f0'].to(device, non_blocking=True) #conditioning
#             # Auto-encode
#             resyn_audio, kl_loss = self(input_data)
#             # Reconstruction loss
#             batch_loss = recon_loss(target, resyn_audio, loss_type=loss_type)
#             test_loss += batch_loss.detach()
#     test_loss /= len(loader)
#     return test_loss

# def render_batch(model, testbatch, base_dir, input_key='mel'):
#     with torch.no_grad():
#         audio = testbatch['audio'].to(device, non_blocking=True)
#         inputdata = testbatch[input_key].to(device, non_blocking=True)
#         resyn_audio, kl = model(inputdata)
#         plot_recons(audio.cpu(), resyn_audio.cpu(), name=plot_dir, epochs=i, num=12)
# nsynth_qualities = ['bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic', 'nonlinear_env', 'percussive', 'reverb', 'tempo-synced']
# nsynth_instruments = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

# sol_instruments = ['Bass Tuba', 'French Horn', 'Trombone', 'Trumpet in C', 'Accordion', 'Cello', 'Contrabass', 'Viola', 'Violin', 'Alto Saxophone', 'Bassoon', 'Clarinet in Bb',  'Flute', 'Oboe']

# instruments = {'nsynth': nsynth_instruments, 'tinysol': sol_instruments}

def videovae_sample(model, device, audio_dir, f0_hz, inst_fam, insts_list, render_batch_size = 6, frame_setting='fine', n_samples=64000, epochs='final'):
    """sample from videovae with some attributes
    model:      ddspsynth model
    f0_hz:      f0 in hz for harmonic synth [n_frames]
    inst_fam (str): ex.) 'bass'
    """
    if not isinstance(model.ae_model, (VideoVAE, CondVideoVAE, AttrRNNLatent, FiLMAttrRNNLatent)):
        raise NotImplementedError
    else:
        f0_hz = f0_hz[None, :, None].expand(render_batch_size, -1, 1).to(device)
        h_0 = torch.rand(render_batch_size, model.ae_model.hidden_size).to(device)*0.01
        inst_idx = insts_list.index(inst_fam)
        label_classes = len(insts_list)
        inst_onehot = torch.eye(label_classes)[inst_idx]
        attributes = inst_onehot
        attributes = attributes[None, :].expand(render_batch_size, -1).to(device)
        _params, audio = model.ae_model.generate(model.synth, h_0, f0_hz, attributes, frame_setting, n_samples)
        save_batch_audio(audio.detach().cpu().numpy(), os.path.join(audio_dir, 'sample_{0}_{1}'.format(inst_fam, epochs)))

def rnnprior_sample(model, device, audio_dir, f0_hz, render_batch_size = 6, frame_setting='fine', n_samples=64000):
    """sample from rnn prior
    model:      ddspsynth model
    f0_hz:      f0 in hz for harmonic synth [n_frames]
    """
    if not isinstance(model.ae_model, (CondRNNPriorVAE, RNNPriorVAE, RNNLatent, FiLMRNNLatent)):
        raise NotImplementedError
    else:
        f0_hz = f0_hz[None, :, None].expand(render_batch_size, -1, 1).to(device)
        h_0 = torch.rand(render_batch_size, model.ae_model.hidden_size).to(device)*0.01
        _params, audio = model.ae_model.generate(model.synth, h_0, f0_hz, frame_setting, n_samples)
        save_batch_audio(audio.detach().cpu().numpy(), os.path.join(audio_dir, 'sample'))

def evaluate_latent_space(model, device, test_loader, test_batch, plot_dir, n_trajectories=6, color_att='inst', n_plots=1000, epochs='final', frame_setting='fine'):
    instruments_list = test_loader.dataset.use_insts
    final_z = []
    att_labels = [] # list of strs ['bass', 'vocal',...]
    with torch.no_grad():
        for data_batch in tqdm.tqdm(test_loader):
            data_batch = {name:tensor.to(device) for name, tensor in data_batch.items()}
            z_tilde, cond = model.det_encode(data_batch)                
            final_z.append(z_tilde)
            if color_att == 'inst': #write labels
                label_classes = len(instruments_list)
                inst_onehots = data_batch['instrument'].unsqueeze(1).expand(-1, z_tilde.shape[1], -1)
                _, inst_labels = torch.max(inst_onehots, -1)
                inst_labels = inst_labels.flatten(0,1) # (batch*n_frames)
                att_labels.extend([il.item() for il in inst_labels])
    cm = plt.cm.get_cmap('hsv')
    final_z = torch.cat(final_z, dim=0)
    final_z = final_z.flatten(0, 1).detach().cpu().numpy()
    pca = decomposition.PCA(n_components=2)
    pca.fit(final_z)
    X = pca.transform(final_z)
    xlim = [min(X[:,0]), max(X[:,0])]
    ylim = [min(X[:,1]), max(X[:,1])]
    pca_file_name = os.path.join(plot_dir, 'pca.pkl')
    with open(pca_file_name, 'wb') as f:
        pickle.dump(pca, f)
    att_labels = np.array(att_labels)
    if color_att:
        latent_scatter(X, plot_dir, n_plots, color_labels=att_labels, epochs=epochs, lims=(xlim,ylim))
    else:
        latent_scatter(X, plot_dir, n_plots, epochs=epochs, lims=(xlim,ylim))
    latent_trajectories(model, test_batch, device, pca, instruments_list, plot_dir, n_trajectories, epochs=epochs, lims=(xlim,ylim))
    latent_feature_map(model, pca, device, plot_dir, epochs=epochs, lims=(xlim,ylim))
    if isinstance(model.ae_model, AttrRNNLatent):
        gen_trajectory(model, pca, device, instruments_list, plot_dir, frame_setting, lims=(xlim,ylim))

def latent_scatter(X, plot_dir, n_plots, color_labels=None, epochs='final', lims=None):
    indices = torch.randperm(X.shape[0])
    X_plot = X[indices[:n_plots]]
    color_labels_plot = color_labels[indices[:n_plots]]
    # plot instrument labels
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_title("latent space scatter plot", size = 14)
    if not lims is None:
        ax1.set_xlim(*lims[0])
        ax1.set_ylim(*lims[1])
    if not color_labels is None:
        scatter = ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=color_labels_plot)
        ax1.legend(*scatter.legend_elements(), loc='lower left', title='instruments')
    else:
        ax1.scatter(X_plot[:, 0], X_plot[:, 1])
    # legend
    fig.savefig(os.path.join(plot_dir, 'latent_space_scatter.png'))

def latent_trajectories(model, data_batch, device, pca, insts_list, plot_dir, n_trajectories=6, epochs='final', lims=None):
    data_batch = {name:tensor[:n_trajectories].to(device) for name, tensor in data_batch.items()}
    z_tilde, cond = model.det_encode(data_batch)
    z_tilde = z_tilde.permute(0,2,1)
    # filtering
    window = signal.gaussian(3, 1)
    kernel = torch.from_numpy(window / window.sum()).float()[None, None, :].to(device)
    kernel = kernel.expand(z_tilde.shape[1], 1, -1)
    draw_z = F.conv1d(z_tilde, kernel, groups=z_tilde.shape[1]).detach().cpu()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_title("latent trajectories", size = 14)
    if not lims is None:
        ax1.set_xlim(*lims[0])
        ax1.set_ylim(*lims[1])
    for i, z_trajectory in enumerate(draw_z):
        z_trajectory = z_trajectory.permute(1, 0)
        X = pca.transform(z_trajectory)
        inst_onehots = cond['instrument']
        _, inst_labels = torch.max(inst_onehots, -1)
        ax1.plot(X[:, 0], X[:, 1], label=insts_list[int(inst_labels[i].item())])
    ax1.legend()
    fig.savefig(os.path.join(plot_dir, 'latent_space_trajectories.png'))

def latent_feature_map(model, pca, device, plot_dir, lims, resolution=100, epochs='final'):
    x = np.linspace(lims[0][0], lims[0][1], resolution)
    y = np.linspace(lims[1][0], lims[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    z = pca.inverse_transform(np.c_[X.ravel(), Y.ravel()])
    index = 0
    scs = []
    sfs = []
    energies = []
    condition = {}
    condition['z'] = torch.from_numpy(z).float().unsqueeze(1).to(device)
    condition['f0_hz'] = torch.ones(condition['z'].shape[0], 1, 1).to(device)*440
    cond = {}
    batch_s = resolution
    n_fft = 2048
    while(True):
        cond['z'] = condition['z'][index:index+batch_s]
        cond['f0_hz'] = condition['f0_hz'][index:index+batch_s]
        with torch.no_grad():
            waveforms = model.decode(cond, n_fft)
        for y in waveforms:
            y = y.detach().cpu().numpy()
            S, phase = librosa.magphase(librosa.stft(y, n_fft=n_fft, center=False))
            scs.append(librosa.feature.spectral_centroid(S=S, sr=16000)[0,0])
            sfs.append(librosa.feature.spectral_flatness(S=S)[0,0])
            energies.append(librosa.feature.rms(S=S)[0,0])
        index+=batch_s
        if index == z.shape[0]:
            break
    # https://qiita.com/kumamupooh/items/c793a6781a753eca6d8e
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    ax1.set_aspect('equal')
    scs_array = np.array(scs).reshape(X.shape)
    mappable = ax1.pcolormesh(X, Y, scs_array, cmap='coolwarm', norm=LogNorm(vmin=400, vmax=2100))
    pp = fig1.colorbar(mappable, ax=ax1, orientation="vertical")
    pp.set_clim(400, 2100)
    pp.set_label('Spectral centroid (Hz)')
    fig1.savefig(os.path.join(plot_dir, 'spectral_centroid_map_{0}.png'.format(epochs)))

    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111)
    ax2.set_aspect('equal')
    sfs_array = np.array(sfs).reshape(X.shape)
    mappable = ax2.pcolormesh(X, Y, sfs_array, cmap='coolwarm', norm=Normalize(vmin=0.0, vmax=1e-2))
    pp = fig2.colorbar(mappable, ax=ax2, orientation="vertical")
    pp.set_clim(0.0, 1e-3)
    pp.set_label('Spectral flatness')
    fig2.savefig(os.path.join(plot_dir, 'spectral_flatness_map_{0}.png'.format(epochs)))

    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111)
    ax3.set_aspect('equal')
    energies_array = np.array(energies).reshape(X.shape)
    mappable = ax3.pcolormesh(X, Y, energies_array, cmap='coolwarm', norm=LogNorm(vmin=1e-5, vmax=1e-1))
    pp = fig3.colorbar(mappable, ax=ax3, orientation="vertical")
    pp.set_clim(1e-5, 1e-1)
    pp.set_label('Spectral energy')
    fig3.savefig(os.path.join(plot_dir, 'spectral_energy_{0}.png'.format(epochs)))

def gen_trajectory(model, pca, device, use_insts, plot_dir, frame_setting, lims=None):
    def prior_trajectories(ae_model, device, inst_fam, insts_list, render_batch_size=16, frame_setting='fine'):
        # set up initial prior with attributes
        h_0 = torch.zeros(render_batch_size, ae_model.hidden_size).to(device)
        n_fft, hop_length = get_window_hop(frame_setting)
        n_frames = math.ceil((64000 - n_fft) / hop_length) + 1
        inst_idx = insts_list.index(inst_fam)
        label_classes = len(insts_list)
        inst_onehot = torch.eye(label_classes)[inst_idx]
        attribute = inst_onehot
        attribute = attribute[None, :].expand(render_batch_size, -1).to(device)
        z = torch.zeros(render_batch_size, n_frames, ae_model.latent_dims).to(device)
        # set up initial prior with attributes
        z_t = torch.zeros(render_batch_size, ae_model.latent_dims).to(device)
        rnn_input = torch.cat([z_t, attribute], dim=-1)
        h = ae_model.temporal(rnn_input, h_0)
        with torch.no_grad():
            for t in range(n_frames):
                # prior distribution with rnn information
                mu_p_t, scale_p_t = ae_model.get_prior(h)    
                z_t = torch.randn_like(mu_p_t) * scale_p_t + mu_p_t
                rnn_input = torch.cat([z_t, attribute], dim=-1)
                h = ae_model.temporal(rnn_input, h)
                z[:,t,:] = z_t
        return z
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    if not lims is None:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
    for inst in use_insts:
        z = prior_trajectories(model.ae_model, device, inst, use_insts, 16, frame_setting)
        window = signal.gaussian(1, 1) # filter
        z_mean = z.mean(dim=0, keepdim=True)
        z_mean = z_mean.permute(0,2,1) # batch, latent, time
        kernel = torch.from_numpy(window / window.sum()).float()[None, None, :].to(device)
        kernel = kernel.expand(z_mean.shape[1], 1, -1)
        draw_z = F.conv1d(z_mean, kernel, groups=z_mean.shape[1]).detach().cpu()
        z_trajectory = z_mean.squeeze(0).permute(1, 0)
        X = pca.transform(z_trajectory.cpu().numpy())
        p = ax.plot(X[:, 0], X[:, 1], label=inst)
    ax.legend()
    fig.savefig(os.path.join(plot_dir, 'prior_trajectory.png'))

def compare_reconstructions(model, test_loader, device, plot_dir, audio_dir, batch_evals=10, render_batch_size = 6):
    # from synth.synthesize import synthesize_batch
    n_evals = 0
    with torch.no_grad():
        for data_batch in test_loader:
            n_evals += 1
            if (n_evals > batch_evals):
                break
            data_batch = {name:tensor[:render_batch_size].to(device) for name, tensor in data_batch.items()}
            output_audio = model(data_batch)
            plot_recons(data_batch['audio'].detach().cpu().numpy(), output_audio.detach().cpu().numpy(), plot_dir, name='evaluate_plot_{0}'.format(n_evals), num=render_batch_size)
            # write audio
            n_samples = data_batch['audio'].shape[-1]
            recon_orig_batch = torch.stack([data_batch['audio'], output_audio], dim=1).view(-1, n_samples).detach().cpu().numpy()
            save_batch_audio(recon_orig_batch, os.path.join(audio_dir, 'compare_recon_{:02}'.format(n_evals)))

def latent_perturbation(model, device, testbatch, audio_dir, render_batch_size = 6, num=8):
    """
    add some noise to the encoding and output audio
    """
    testbatch = {name:tensor.to(device) for name, tensor in testbatch.items()}
    z, conditioning = model.encode(testbatch)
    n_frames = z.shape[1]
    latent_size = z.shape[2]
    for b in range(render_batch_size):
        # grab particular sample from batch and expand along batch dim
        cond = {name:tensor[b].expand(num, *tensor.shape[1:]).to(device) for name, tensor in conditioning.items()}
        # original and noisy sequences
        noise = torch.cat([torch.zeros(1, latent_size), torch.randn(num-1, latent_size)*0.5])[:, None, :].expand(num, n_frames, latent_size).to(device)
        cond['z'] = noise + cond['z']
        output_audio = model.decode(cond)
        # latent_neightborhood
        save_batch_audio(output_audio.detach().cpu().numpy(), os.path.join(audio_dir, 'latent_perturb_{:02}'.format(b)))

def latent_interp(model, device, testbatch, audio_dir, num = 6, n_interp = 8):
    """
    interpolates between z_t of two sounds
    """
    testbatch = {name:tensor.to(device) for name, tensor in testbatch.items()}
    z_tilde, conditioning = model.encode(testbatch)
    z = conditioning['z']
    batch_size, n_frames, latent_size = z.shape
    with torch.no_grad():
        for b in range(num):
            index_1, index_2 = np.random.choice(batch_size, 2, replace=False)
            # grab particular sample from batch and expand along batch dim
            cond_1 = {name:tensor[index_1].expand(n_interp, *tensor.shape[1:]).to(device) for name, tensor in conditioning.items()}
            cond_2 = {name:tensor[index_2].expand(n_interp, *tensor.shape[1:]).to(device) for name, tensor in conditioning.items()}
            # original and noisy sequences
            z_1 = cond_1['z']
            z_2 = cond_2['z']
            z_interp = torch.linspace(0, 1.0, n_interp, device=device)[:, None, None] * z_1 + torch.linspace(1.0, 0, n_interp, device=device)[:, None, None] * z_2
            # just use the conditions for the first one I guess
            cond_1['z'] = z_interp
            output_audio_1 = model.decode(cond_1)
            save_batch_audio(output_audio_1.detach().cpu().numpy(), os.path.join(audio_dir, 'latent_interp_{:02}_{:02}_a'.format(index_1, index_2)))
            cond_2['z'] = z_interp
            output_audio_2 = model.decode(cond_2) 
            save_batch_audio(output_audio_2.detach().cpu().numpy(), os.path.join(audio_dir, 'latent_interp_{:02}_{:02}_b'.format(index_1, index_2)))

def save_batch_audio(audio, name, sr=16000):
    """
    save a batch of audio into single file
    audio: numpy array [n_batch, n_samples]
    """
    # Figure out len of full audio
    final_size = sum([int(f.shape[0]) for f in audio]) + (sr*0.1 * len(audio))
    wave_out = np.zeros(int(final_size))
    cur_p = 0
    for b in range(len(audio)):
        wave_out[cur_p:int(cur_p + audio[b].shape[0])] += audio[b]
        cur_p += int(audio[b].shape[0] + sr*0.1)
    sf.write(name + '.wav', wave_out, sr)


def save_losses_csv(losses, name):
    """
    save the losses into csv
    losses: detached cpu tensor of losses [epochs, types_of_losses]
    """
    with open(name, 'w') as f:
        if losses.shape[1] == 4:
            fieldnames = ['Epoch', 'Train_loss', 'Train_z_loss', 'Valid_loss', 'Valid_ld_loss']
            dw = csv.DictWriter(f, fieldnames=fieldnames)
            dw.writeheader()
            for i, l in enumerate(losses):
                losses_dict = {'Epoch': i+1, 'Train_loss': l[0].item(), 'Train_z_loss': l[1].item(), 
                                'Valid_loss': l[2].item(), 'Valid_ld_loss': l[3].item()}
                dw.writerow(losses_dict)
        if losses.shape[1] == 2: # Classifier
            fieldnames = ['Epoch', 'Train_loss', 'Valid_accuracy']
            dw = csv.DictWriter(f, fieldnames=fieldnames)
            dw.writeheader()
            for i, l in enumerate(losses):
                losses_dict = {'Epoch': i+1, 'Train_loss': l[0].item(), 'Valid_accuracy': l[1].item()*100}
                dw.writerow(losses_dict)

def test_losses(model, test_loader, device, plot_dir):
    total_mse = 0
    total_lsd = 0
    print('calculating losses on test set')
    with torch.no_grad():
        for data_batch in test_loader:
            data_batch = {name:tensor.to(device) for name, tensor in data_batch.items()}
            orig_audio = data_batch['audio']
            resyn_audio = model(data_batch)
            orig_power_s = spectrogram(orig_audio).detach()
            resyn_power_s = spectrogram(resyn_audio).detach()
            # http://rug.mnhn.fr/seewave/HTML/MAN/logspec.dist.html
            lsd = torch.sqrt(torch.sum((10*torch.log10(orig_power_s/resyn_power_s + 1e-5))**2, dim=1))
            lsd = lsd.mean()
            mse = F.mse_loss(orig_power_s, resyn_power_s, reduction='none')
            mse = mse.mean()
            total_mse += mse.detach()
            total_lsd += lsd.detach()
    total_mse /= len(test_loader)
    total_lsd /= len(test_loader)
    result_str = 'MSE: {0:.5f}, LSD: {1:.5f}'.format(total_mse, total_lsd)
    with open(os.path.join(plot_dir, 'testloss.txt'), 'w') as f:
        f.write(result_str)
    print(result_str)
    