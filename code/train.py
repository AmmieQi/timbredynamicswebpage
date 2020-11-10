import torch
import torch.optim as optim
import torchaudio
import numpy as np
import os, argparse, pickle, warnings, json
import tqdm
from ddspsynth.modelutils import construct_encoder_decoder, construct_synth, construct_ae_model
from ddspsynth.data import load_nsynth_dataset, load_filtnsynth_dataset, load_tinysol_dataset
from ddspsynth.spectral import SpectralLoss
from ddspsynth.plot import plot_recons
from ddspsynth.evaluate import save_losses_csv, videovae_sample, evaluate_latent_space
from ddspsynth.ddspmodels import DDSPSynth
from torch.autograd import detect_anomaly
from torch.utils.data import DataLoader, Subset
from ddspsynth.videovae import VideoVAE, RNNPriorVAE, CondRNNPriorVAE, CondVideoVAE
from ddspsynth.dynamicaes import AttrRNNLatent

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output',           type=str,                           help='')
    parser.add_argument('data_path',        type=str,                           help='')
    parser.add_argument('--dataset',        type=str,   default='fnsynth',      help='')
    parser.add_argument('--data_filt',      type=str,   default='a',            help='')
    parser.add_argument('--use_quality',    action='store_true')

    # Model arguments
    parser.add_argument('--synth',          type=str,   default='hpn',          help='')
    parser.add_argument('--model',          type=str,   default='mlp_mlp',     help='')
    parser.add_argument('--ae',             type=str,   default='vae',          help='')
    parser.add_argument('--latent_size',    type=int,   default=64,             help='')
    #encoder
    parser.add_argument('--encoder_dims',   type=int,   default=64,             help='')
    parser.add_argument('--z_steps',        type=str,   default='coarse',       help='')
    parser.add_argument('--n_mels',         type=int,   default=40,       help='')
    # wave encoder
    parser.add_argument('--enc_channels',   type=int,   default=32,             help='')
    parser.add_argument('--enc_kernel',     type=int,   default=15,             help='')
    parser.add_argument('--enc_strides',    type=int,   default=[2,4,4,4],      nargs='*', help='')
    #decoder
    parser.add_argument('--use_f0',         action='store_true')
    parser.add_argument('--use_ld',         action='store_true')

    parser.add_argument('--clip',           type=float, default=5.0,            help='')
    parser.add_argument('--loss',           type=str,   default='L1',           help='')
    parser.add_argument('--fft_sizes',      type=int,   default=[64, 128, 256, 512, 1024, 2048], nargs='*', help='')
    parser.add_argument('--beta_factor',    type=float, default=1.0,            help='')
    parser.add_argument('--start_latent',   type=int,   default=20,              help='')
    parser.add_argument('--warm_latent',    type=int,   default=100,            help='')


    # Optimization arguments
    parser.add_argument('--batch_size',     type=int,   default=64,             help='')
    parser.add_argument('--epochs',         type=int,   default=200,            help='')
    parser.add_argument('--lr',             type=float, default=2e-4,           help='')
    parser.add_argument('--plot_interval',  type=int,   default=10,             help='')
    parser.add_argument('--subset',         type=int,   default=None,           help='')
    # performance related arguments
    parser.add_argument('--device',         type=str,   default='cuda',         help='Device for CUDA')
    parser.add_argument('--nbworkers',      type=int,   default=4,              help='')

    args = parser.parse_args()
    torchaudio.set_audio_backend("sox_io")

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    audio_dir = os.path.join(args.output, 'audio')
    plot_dir = os.path.join(args.output, 'plot')
    model_dir = os.path.join(args.output, 'model')
    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)
        os.mkdir(plot_dir)
        os.mkdir(model_dir)
    
    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Optimization will be on ' + str(device) + '.')

    # load dataset
    # precomputed features
    # feature_settings = {'f0':{'normalize':False}} 
    suffix = args.data_filt if args.dataset == 'fnsynth' else ''
    dataset_file = os.path.join(args.data_path, 'datasets_{0}.pkl'.format(suffix))
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            dset_train, dset_valid, dset_test = pickle.load(f)
    else:
        if args.dataset == 'nsynth':
            dset_train, dset_valid, dset_test = load_nsynth_dataset(args.data_path)
        elif args.dataset == 'fnsynth':
            dset_train, dset_valid, dset_test = load_filtnsynth_dataset(args.data_path, args.data_filt, args.use_quality)
        elif args.dataset == 'tinysol': 
            dset_train, dset_valid, dset_test = load_tinysol_dataset(args.data_path)
        with open(dataset_file, 'wb') as f:
            pickle.dump([dset_train, dset_valid, dset_test], f)

    dset_train.use_quality = args.use_quality
    dset_valid.use_quality = args.use_quality
    dset_test.use_quality = args.use_quality
    if args.subset:
        indices = np.random.choice(len(dset_train), args.subset, replace=False)
        dset_train = Subset(dset_train, indices)

    dl_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.nbworkers, pin_memory=False)
    dl_valid = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=False)
    dl_test = DataLoader(dset_test, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=False)
    
    testbatch = next(iter(dl_test))
    test_f0 = testbatch['f0_hz'][10].clone()
    print('Loaded dataset')
    print('Size of training dataset: '+str(len(dset_train)))

    # making the synth
    synth = construct_synth(args.synth)

    encoder, decoder = construct_encoder_decoder(args, synth.ext_param_size)

    ae_model = construct_ae_model(encoder, decoder, args, testbatch['attributes'].shape[-1])
    # create final model
    model = DDSPSynth(ae_model, synth).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters of the model: {0}'.format(total_params))
    recon_loss = SpectralLoss(args.fft_sizes, log_mag_w=1.0, mag_w=1.0)

    with open(os.path.join(args.output, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-7)
    # get number of frames
    testbatch = {name:tensor.to(device) for name, tensor in testbatch.items()}
    z_frames, f_frames, ld_frames = model.get_n_frames(testbatch)
    print('Frames: z:{0} f0:{1} loudness:{2}'.format(z_frames, f_frames, ld_frames))
    #% Monitoring quantities
    losses = torch.zeros(args.epochs, 4)
    best_loss = np.inf
    print('[Starting training]')
    with tqdm.tqdm(range(args.epochs)) as pbar:
        for i in pbar:
            l = max(i - args.start_latent, 0)
            beta = args.beta_factor * (float(l) / float(max(args.warm_latent, l)))
            losses[i, 0], losses[i, 1] = model.train_epoch(loader=dl_train, recon_loss=recon_loss, optimizer=optimizer, device=device, clip=args.clip, loss_type=args.loss, beta=beta)
            losses[i, 2], losses[i, 3] = model.eval_epoch(loader=dl_valid, recon_loss=recon_loss, device=device, loss_type=args.loss)
            tqdm.tqdm.write('Epoch: {0:03} Tr. Loss: {1:.5f} Val. Loss: {2:.5f} Val. Ld Loss: {3:.3f} Z loss: {4:.5f} Beta: {5:.4f}'.format(i+1, losses[i, 0], losses[i, 2], losses[i, 3], losses[i, 1], beta))
            save_losses_csv(losses, os.path.join(args.output, 'losses.csv'))
            if i > args.warm_latent:
                scheduler.step(losses[i, 2])
            if (losses[i, 2] < best_loss):
                # Save model
                best_loss = losses[i, 2]
                torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict.pth'))
            if (i+1) % args.plot_interval == 0:
                # plot spectrograms
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    resyn_audio = model(testbatch)
                    plot_recons(testbatch['audio'].detach().cpu().numpy(), resyn_audio.detach().cpu().numpy(), plot_dir=plot_dir, epochs=i+1, num=12)
                    # if isinstance(model.ae_model, (VideoVAE, CondVideoVAE, AttrRNNLatent)):
                    #     insts_list = dl_test.dataset.use_insts
                    #     videovae_sample(model, device, audio_dir, test_f0, 'string', insts_list, frame_setting=args.z_steps, epochs=i+1)
                    #     videovae_sample(model, device, audio_dir, test_f0, 'flute', insts_list, frame_setting=args.z_steps, epochs=i+1)
                    #     videovae_sample(model, device, audio_dir, test_f0, 'brass', insts_list, frame_setting=args.z_steps, epochs=i+1)
                    #     evaluate_latent_space(model, device, dl_test, testbatch, plot_dir, color_att='inst', epochs=i+1)
                torch.save(model.state_dict(), os.path.join(model_dir, 'state_dict_epoch{:03}.pth'.format(i+1)))