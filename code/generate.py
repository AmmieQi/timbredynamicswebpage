import argparse, os, glob, warnings
from librosa.feature import melspectrogram
import librosa
import numpy as np
import torch
from ddspsynth.util import pad_or_trim_to_expected_length
import tqdm
from ddspsynth.spectral import compute_f0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str, help='')
    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args()

    warnings.simplefilter('ignore', RuntimeWarning) # suppress invalid value error
    audio_dir = os.path.join(args.base_dir, 'audio')
    raws = glob.glob(os.path.join(audio_dir,'*.wav'))

    mel_dir = os.path.join(args.base_dir, 'mel')
    if not os.path.exists(mel_dir):
        os.mkdir(mel_dir)
    # multifft_dir = os.path.join(args.base_dir, 'multifft')
    # if not os.path.exists(multifft_dir):
    #     os.mkdir(multifft_dir)
    f0_dir = os.path.join(args.base_dir, 'f0')
    if not os.path.exists(f0_dir):
        os.mkdir(f0_dir)

    for raw_file in tqdm.tqdm(raws):
        file_name = os.path.splitext(os.path.basename(raw_file))[0]
        audio, sr = librosa.load(raw_file, sr=16000)
        mel_file = os.path.join(mel_dir, file_name+'.npy')
        f0_file = os.path.join(f0_dir, file_name+'.npy')
        if not os.path.exists(mel_file):
            mel = melspectrogram(audio, sr=args.sr, n_mels=40, n_fft=2048, hop_length=1024, fmin=30)
            np.save(mel_file, mel)
        if not os.path.exists(f0_file):
            f0 = compute_f0(audio, sample_rate=args.sr, frame_rate=20)
            np.save(f0_file, f0)
            

