from ddspsynth.modules.generators import Additive, FilteredNoise, Sinusoids
from ddspsynth.synth import Synthesizer, Add
from ddspsynth.util import exp_sigmoid
from ddspsynth.encoders import DDSPEncoder, MelMlpEncoder, WaveEncoder, MelConvEncoder, MfccMlpEncoder
from ddspsynth.decoders import DDSPDecoder, MlpDecoder
from ddspsynth.aes import AE, VAE
from ddspsynth.videovae import VideoVAE, RNNPriorVAE, CondRNNPriorVAE, CondVideoVAE
from ddspsynth.dynamicaes import RNNLatent, AttrRNNLatent, FiLMRNNLatent, FiLMAttrRNNLatent

def construct_synth(synth_name):
    if synth_name == 'additive_f0':
        harmonic = Additive(n_harmonics=128)
        dag = [(harmonic, {'amplitudes': 'AMP', 'harmonic_distribution': 'HARM', 'f0_hz': 'f0_hz'})]
        synth = Synthesizer(dag, fixed_params={'f0_hz':None})
    elif synth_name == 'additive':
        harmonic = Additive(n_harmonics=128, scale_fn=exp_sigmoid)
        dag = [(harmonic, {'amplitudes': 'AMP', 'harmonic_distribution': 'HARM', 'f0_hz': 'F0'})]
        synth = Synthesizer(dag)
    elif synth_name == 'noise':
        noise = FilteredNoise(filter_size=2048, initial_bias=-5.0, amplitude=8.0)
        dag = [(noise, {'freq_response': 'FIL'})]
        synth = Synthesizer(dag)
    elif synth_name == 'sinusoids':
        sinusoids = Sinusoids(name='sin0')
        noise = FilteredNoise(name='noise0')
        add = Add(name='add')
        dag = [(sinusoids, {'amplitudes': 'AMP', 'frequencies': 'FREQ'}) ,(noise, {'freq_response': 'FIL'}), (add, {'signal_a': 'sin0', 'signal_b': 'noise0'})]
        synth = Synthesizer(dag, fixed_params={})
    elif synth_name == 'hpn':
        harmonic = Additive(name='harm0', scale_fn=exp_sigmoid)
        noise = FilteredNoise(name='noise0', scale_fn=exp_sigmoid, initial_bias=-5.0, amplitude=2.0)
        add = Add(name='add')
        dag = [(harmonic, {'amplitudes': 'AMP', 'harmonic_distribution': 'HARM', 'f0_hz': 'f0_hz'}),(noise, {'freq_response': 'FIL'}), (add, {'signal_a': 'harm0', 'signal_b': 'noise0'})]
        synth = Synthesizer(dag, fixed_params={'f0_hz':None})
    
    return synth

def construct_encoder_decoder(args, synth_param_size):
    # create encoder/decoder
    enc_type, dec_type = args.model.lower().split('_')
    if enc_type == 'wave':
        encoder = WaveEncoder(args.z_steps, args.encoder_dims, args.enc_channels, args.enc_kernel, args.enc_strides, encode_ld=True)
    elif enc_type == 'mlp':
        encoder = MelMlpEncoder(args.z_steps, args.encoder_dims, args.n_mels, encode_ld=args.use_ld)
    elif enc_type == 'conv':
        encoder = MelConvEncoder(args.z_steps, args.encoder_dims, args.n_mels, encode_ld=args.use_ld)
    elif enc_type == 'ddsp':
        encoder = DDSPEncoder(args.z_steps, args.encoder_dims, args.n_mels, encode_ld=args.use_ld)
    elif enc_type == 'mmlp':
        encoder = MfccMlpEncoder(args.z_steps, args.encoder_dims, args.n_mels, encode_ld=args.use_ld)
    else:
        raise ValueError('Wrong encoder type')
    if dec_type == 'mlp':
        decoder = MlpDecoder(args.latent_size, synth_param_size, use_f0=args.use_f0, use_ld=args.use_ld)
    elif dec_type == 'ddsp':
        decoder = DDSPDecoder(args.latent_size, synth_param_size, use_f0=args.use_f0, use_ld=args.use_ld)
    else:
        raise ValueError('Wrong decoder type')

    return encoder, decoder

def construct_ae_model(encoder, decoder, args, att_size=None):
    # create autoencoder
    if args.ae == 'vae':
        ae_model = VAE(encoder, decoder, args.encoder_dims, args.latent_size)
    elif args.ae == 'ae':
        args.warm_latent=1
        ae_model = AE(encoder, decoder, args.encoder_dims, args.latent_size)
    elif args.ae == 'videovae':
        ae_model = VideoVAE(encoder, decoder, args.encoder_dims, args.latent_size, att_size)
    elif args.ae == 'rnnprior':
        ae_model = RNNPriorVAE(encoder, decoder, args.encoder_dims, args.latent_size)
    elif args.ae == 'condrnnprior':
        ae_model = CondRNNPriorVAE(encoder, decoder, args.encoder_dims, args.latent_size)
    elif args.ae == 'condvideovae':
        ae_model = CondVideoVAE(encoder, decoder, args.encoder_dims, args.latent_size, att_size)
    elif args.ae == 'rnnlatent':
        ae_model = RNNLatent(encoder, decoder, args.encoder_dims, args.latent_size)
    elif args.ae == 'attrrnn':
        ae_model = AttrRNNLatent(encoder, decoder, args.encoder_dims, args.latent_size, att_size)
    elif args.ae == 'filmrnnlatent':
        ae_model = FiLMRNNLatent(encoder, decoder, args.encoder_dims, args.latent_size, att_size)
    elif args.ae == 'filmattrrnn':
        ae_model = FiLMAttrRNNLatent(encoder, decoder, args.encoder_dims, args.latent_size, att_size)
    return ae_model