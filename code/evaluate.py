import argparse, json, os, pickle
import torch
from ddspsynth.evaluate import compare_reconstructions, latent_perturbation, latent_interp, rnnprior_sample, videovae_sample, evaluate_latent_space, test_losses
from ddspsynth.ddspmodels import DDSPSynth
from ddspsynth.aes import AE, VAE
from ddspsynth.videovae import VideoVAE, RNNPriorVAE, CondRNNPriorVAE, CondVideoVAE
from ddspsynth.modelutils import construct_encoder_decoder, construct_synth, construct_ae_model
from torch.utils.data import DataLoader
from types import SimpleNamespace
from mpl_toolkits.mplot3d import Axes3D
from ddspsynth.dynamicaes import AttrRNNLatent, RNNLatent, FiLMAttrRNNLatent, FiLMRNNLatent

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='')
    parser.add_argument('--device', type=str, default='cuda', help='Device for CUDA')

    eval_args = parser.parse_args()

    audio_dir = os.path.join(eval_args.output, 'audio')
    plot_dir = os.path.join(eval_args.output, 'plot')
    model_dir = os.path.join(eval_args.output, 'model/state_dict.pth')
    args_file = os.path.join(eval_args.output, 'args.txt')
    with open(args_file) as f:
        args = json.load(f)
        args = SimpleNamespace(**args)
    
    # set device
    device = torch.device(eval_args.device if torch.cuda.is_available() else 'cpu')
    print('Evaluation will be on ' + str(device) + '.')

    # load dataset
    suffix = args.data_filt if args.dataset == 'fnsynth' else ''
    dataset_file = os.path.join(args.data_path, 'datasets_{0}.pkl'.format(suffix))
    with open(dataset_file, 'rb') as f:
        dset_train, dset_valid, dset_test = pickle.load(f)
    
    dset_train.use_quality = args.use_quality
    dset_valid.use_quality = args.use_quality
    dset_test.use_quality = args.use_quality
    dl_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.nbworkers, pin_memory=True)
    dl_valid = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=True)
    dl_test = DataLoader(dset_test, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=True)
    
    testbatch = next(iter(dl_test))

    print('Loaded dataset')

    # recreate model
    synth = construct_synth(args.synth)

    encoder, decoder = construct_encoder_decoder(args, synth.ext_param_size)

    # create autoencoder
    ae_model = construct_ae_model(encoder, decoder, args, testbatch['attributes'].shape[-1])

    # create final model
    model = DDSPSynth(ae_model, synth).to(device)

    model.load_state_dict(torch.load(model_dir))
    model.eval()

    print('[Starting evaluation]')

    f0 = torch.ones(100).to(device)*440.0
    if isinstance(model.ae_model, (VideoVAE, CondVideoVAE, AttrRNNLatent, FiLMAttrRNNLatent)):
        print('Sampling from videovae...')
        insts_list = dl_test.dataset.use_insts
        for inst in insts_list:
            videovae_sample(model, device, audio_dir, f0, inst, insts_list, frame_setting=args.z_steps)
    elif isinstance(model.ae_model, (RNNPriorVAE, CondRNNPriorVAE, RNNLatent, FiLMRNNLatent)):
        print('Sampling from rnn prior...')
        rnnprior_sample(model, device, audio_dir, f0, frame_setting=args.z_steps)
    print('Performing reconstructions')
    compare_reconstructions(model, dl_test, device, plot_dir, audio_dir, 6)
    # # TODO: inception score
    print('Evaluating latent space')
    evaluate_latent_space(model, device, dl_test, testbatch, plot_dir, color_att='inst', frame_setting=args.z_steps)
    test_losses(model, dl_test, device, plot_dir)
    # print('Perturbing latent')
    # latent_perturbation(model, device, testbatch, audio_dir)
    # print('Performing latent interpolation')
    # latent_interp(model, device, testbatch, audio_dir)