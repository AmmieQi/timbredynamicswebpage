import os, json, glob, copy, csv, pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize
import torchaudio
import librosa
from scipy.io.wavfile import read as readwav
import torch.nn.functional as F

# import torchcrepe
import tqdm
from ddspsynth.spectral import compute_f0
from ddspsynth.util import pad_or_trim_to_expected_length
class FeatureData(Dataset):
    """Single feature data"""
    def __init__(self, feature, feature_files, normalize=True, transforms=None, set_type=None, train_noise=False, stats=None):
        self.feature = feature
        # Spectral transforms
        self.feature_files = feature_files
        # Retrieve list of files
        # Compute mean and std of dataset
        self.transforms = []
        self.normalize = normalize
        if transforms:
            self.transforms.extend(transforms)
        if normalize:
            if stats:
                self.mean, self.stdev = stats
            else:
                self.compute_normalization()
                
        # TODO:apply noise to training data if set_type=='train' and train_noise
        self.transforms = Compose(self.transforms)

    def compute_normalization(self):
        self.mean = 0
        self.var = 0
        # Parse dataset to compute mean and norm
        tr = Compose(self.transforms)
        for n in range(len(self.feature_files)):
            data = np.load(self.feature_files[n], allow_pickle=True)
            data = torch.from_numpy(data).float()
            data = tr(data) # apply transforms
            # Current file stats
            b_mean = data.mean()
            b_var = (data - self.mean)
            # Running mean and var
            self.mean = self.mean + ((b_mean - self.mean) / (n + 1))
            self.var = self.var + ((data - self.mean) * b_var).mean()
        self.mean = float(self.mean)
        self.var = float(self.var / len(self.feature_files))
        self.stdev = float(np.sqrt(self.var / len(self.feature_files)))

    def __getitem__(self, idx):
        data = np.load(self.feature_files[idx])
        data = torch.from_numpy(data).float()
        data = self.transforms(data)
        if self.normalize:
            data = (data - self.mean) / self.stdev
        return data

    def __len__(self):
        return len(self.feature_files)    

class SoundDataset(Dataset):
    """
    raw audio (wav) / features (melspectrogram, f0, loudness, etc.)
    base_dir/audio/some_random_sound_001.wav
            /{feature_name}/some_random_sound_001.npy
    
    feature_settings = {'f0':{'normalize':False}, 'melspectrogram':{'normalize'=True}}
    """
    def __init__(self, base_dir, feature_settings, raw_file_type='.wav', splits=[.8, .1, .1], shuffle_files=True, sample_rate=16000):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'audio')
        self.feature_settings = feature_settings
        self.feat_dir = {}
        self.feat_files = {}
        # Retrieve list of files
        self.raw_files = sorted(glob.glob(os.path.join(self.raw_dir, '*'+raw_file_type)))
        self.file_names = [os.path.splitext(os.path.basename(t))[0] for t in self.raw_files]

        for feat in feature_settings:
            self.feat_dir[feat] = os.path.join(base_dir, feat)
            assert os.path.exists(self.feat_dir[feat])
            self.feat_files[feat] = [os.path.join(self.feat_dir[feat], fn) + '.npy' for fn in self.file_names]

        self.sample_rate = sample_rate
        # Analyze dataset
        self.analyze_dataset()
        # Create splits
        self.create_splits(splits, shuffle_files)
        self.switch_set('train')

    def analyze_dataset(self):
        raw_file = self.raw_files[0]
        y, sr = librosa.load(raw_file, sr=None, mono=True)
        self.orig_sample_rate = sr
        self.max_length = len(y) / sr #seconds        

    def create_splits(self, splits, shuffle_files):
        nb_files = len(self.raw_files)
        if (shuffle_files):
            idx = np.random.permutation(nb_files)
        else:
            idx = np.linspace(0, nb_files-1, nb_files).astype('int')
        train_idx = idx[:int(splits[0]*nb_files)]
        valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
        test_idx = idx[int((splits[0]+splits[1])*nb_files):]

        self.train_raws = [self.raw_files[i] for i in train_idx]
        self.valid_raws = [self.raw_files[i] for i in valid_idx]
        self.test_raws = [self.raw_files[i] for i in test_idx]
        self.train_feat_files={}
        self.valid_feat_files={}
        self.test_feat_files={}
        self.train_dataset={}
        self.valid_dataset={}
        self.test_dataset={}
        self.means = {}
        self.stdevs = {}
        for feat in self.feature_settings:
            self.train_feat_files[feat] = [self.feat_files[feat][i] for i in train_idx]
            self.valid_feat_files[feat] = [self.feat_files[feat][i] for i in valid_idx]
            self.test_feat_files[feat] = [self.feat_files[feat][i] for i in test_idx]
            self.train_dataset[feat] = FeatureData(feat, self.train_feat_files[feat], **self.feature_settings[feat], set_type='train')
            if self.feature_settings[feat]['normalize']:
                # save the mean/stdev of training set
                self.means[feat] = self.train_dataset[feat].mean
                self.stdevs[feat] = self.train_dataset[feat].stdev
                self.valid_dataset[feat] = FeatureData(feat, self.valid_feat_files[feat], **self.feature_settings[feat], set_type='valid', stats=[self.means[feat], self.stdevs[feat]])
                self.test_dataset[feat] = FeatureData(feat, self.test_feat_files[feat], **self.feature_settings[feat], set_type='test', stats=[self.means[feat], self.stdevs[feat]])
            else:
                self.valid_dataset[feat] = FeatureData(feat, self.valid_feat_files[feat], **self.feature_settings[feat], set_type='valid')
                self.test_dataset[feat] = FeatureData(feat, self.test_feat_files[feat], **self.feature_settings[feat], set_type='test')

    def switch_set(self, name):
        if name == 'train':
            self.curr_dataset = self.train_dataset
            self.curr_raws = self.train_raws
        if name == 'valid':
            self.curr_dataset = self.valid_dataset
            self.curr_raws = self.valid_raws
        if name == 'test':
            self.curr_dataset = self.test_dataset
            self.curr_raws = self.test_raws
        return self

    def __getitem__(self, idx):
        output = {}
        output['audio'], _sr = librosa.load(self.curr_raws[idx], sr=self.sample_rate, duration=self.max_length)
        for feat in self.feature_settings:
            output[feat] = self.curr_dataset[feat][idx]
        return output

    def __len__(self):
        return len(self.curr_raws)

class AudioDataset(Dataset):
    """
    Just retrieve audio from a folder with bunch of raw audio (wav) 
    """
    def __init__(self, base_dir, raw_file_type='.wav', splits=[.8, .1, .1], shuffle_files=True, sample_rate=16000):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'audio')
        # Retrieve list of files
        self.raw_files = sorted(glob.glob(os.path.join(self.raw_dir, '*'+raw_file_type)))
        self.file_names = [os.path.splitext(os.path.basename(t))[0] for t in self.raw_files]

        self.sample_rate = sample_rate
        # Analyze dataset
        self.analyze_dataset()
        # Create splits
        self.create_splits(splits, shuffle_files)
        self.switch_set('train')

    def analyze_dataset(self):
        raw_file = self.raw_files[0]
        y, sr = librosa.load(raw_file, sr=None, mono=True)
        self.orig_sample_rate = sr
        self.max_length = len(y) / sr #seconds        

    def create_splits(self, splits, shuffle_files):
        nb_files = len(self.raw_files)
        if (shuffle_files):
            idx = np.random.permutation(nb_files)
        else:
            idx = np.linspace(0, nb_files-1, nb_files).astype('int')
        train_idx = idx[:int(splits[0]*nb_files)]
        valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
        test_idx = idx[int((splits[0]+splits[1])*nb_files):]

        self.train_raws = [self.raw_files[i] for i in train_idx]
        self.valid_raws = [self.raw_files[i] for i in valid_idx]
        self.test_raws = [self.raw_files[i] for i in test_idx]

    def switch_set(self, name):
        if name == 'train':
            self.curr_raws = self.train_raws
        if name == 'valid':
            self.curr_raws = self.valid_raws
        if name == 'test':
            self.curr_raws = self.test_raws
        return self

    def __getitem__(self, idx):
        output = {}
        output['audio'], _sr = librosa.load(self.curr_raws[idx], sr=self.sample_rate, duration=self.max_length)
        return output

    def __len__(self):
        return len(self.curr_raws)

class NsynthDataset(Dataset):
    """
    Retrieve Nsynth audio and attributes and precomputed f0
    """
    def __init__(self, base_dir, splits=[.8, .1, .1], shuffle_files=True, sample_rate=16000):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'audio')
        self.f0_dir = os.path.join(base_dir, 'f0')
        with open(os.path.join(self.base_dir, 'examples.json')) as f:
            json_dict = json.load(f)
        # Retrieve list of files
        self.json_notes = list(json_dict.values())
        self.note_names = list(json_dict.keys())

        self.sample_rate = sample_rate
        # Analyze dataset
        self.analyze_dataset()
        # Create splits
        self.create_splits(splits, shuffle_files)
        self.switch_set('train')

    def analyze_dataset(self):
        raw_file = os.path.join(self.raw_dir, self.note_names[0]+'.wav')
        y, sr = librosa.load(raw_file, sr=None, mono=True)
        self.orig_sample_rate = sr
        self.max_length = len(y) / sr #seconds        

    def create_splits(self, splits, shuffle_files):
        nb_files = len(self.note_names)
        if (shuffle_files):
            idx = np.random.permutation(nb_files)
        else:
            idx = np.linspace(0, nb_files-1, nb_files).astype('int')
        self.train_idx = idx[:int(splits[0]*nb_files)]
        self.valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
        self.test_idx = idx[int((splits[0]+splits[1])*nb_files):]

    def switch_set(self, name):
        if name == 'train':
            self.curr_idx = self.train_idx
        if name == 'valid':
            self.curr_idx = self.valid_idx
        if name == 'test':
            self.curr_idx = self.test_idx
        return self

    def __getitem__(self, idx):
        output = {}
        index = self.curr_idx[idx]
        note = self.json_notes[index]
        file_name = os.path.join(self.raw_dir, note['note_str']+'.wav')
        inst_fam = note['instrument_family']
        inst_onehot = torch.eye(11)[int(inst_fam)]
        quality_vec = torch.Tensor(note['qualities'])
        attributes = torch.cat([inst_onehot, quality_vec])
        # f0
        f0_file_name = os.path.join(self.f0_dir, note['note_str']+'.npy')
        f0 = np.load(f0_file_name)
        f0 = torch.from_numpy(f0).float()
        output['audio'], _sr = librosa.load(file_name, sr=self.sample_rate, duration=self.max_length)
        output['attributes'] = attributes
        output['f0_hz'] = f0
        return output

    def __len__(self):
        return len(self.curr_idx)

class FilteredNsynthDataset(Dataset):
    def __init__(self, base_dir, filter_args, splits=[.8, .1, .1], shuffle_files=True, sample_rate=16000, length=4.0, use_quality=False):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'audio')
        self.f0_file_path = os.path.join(base_dir, 'f0s.pkl')
        with open(os.path.join(self.base_dir, 'examples.json')) as f:
            self.json_dict = json.load(f)
        # Retrieve list of files
        self.sample_rate = sample_rate
        self.length = 4.0
        self.use_quality = use_quality
        self.filter_dataset(**filter_args)
        self.f0s = {} #dict of np arrays 
        if os.path.exists(self.f0_file_path):
            with open(self.f0_file_path, 'rb') as f:
                self.f0s = pickle.load(f)
        # Analyze dataset
        self.analyze_dataset()
        # Create splits
        self.create_splits(splits, shuffle_files)
        self.switch_set('train')

    def filter_dataset(self, use_insts, ng_source_list, ng_quality_list, pitch_lower=48, pitch_upper=72):
        # same as GANSynth
        self.insts_dict = {inst: k for k, inst in enumerate(use_insts)}
        self.use_insts = use_insts
        self.ng_source_list = ng_source_list
        self.ng_quality_list = ng_quality_list
        # self.pitch_range = range(28, 85)
        self.pitch_range = range(pitch_lower, pitch_upper)
        def filt(entry):
            if entry['instrument_source_str'] in ng_source_list:
                return False
            elif not entry['pitch'] in self.pitch_range:
                return False
            elif not entry['instrument_family_str'] in use_insts:
                return False
            elif any([(q in entry['qualities_str']) for q in ng_quality_list]):
                return False
            else:
                return True
        self.filtered_dict = {key:entry for key, entry in self.json_dict.items() if filt(entry)}
        print('{0} files after filtering'.format(len(self.filtered_dict)))
        self.filtered_keys = list(self.filtered_dict.keys())
    
    def analyze_dataset(self):
        # create f0 or not
        # check number of f0s
        if len(self.f0s) < len(self.filtered_keys):
            self.generate_f0s()

    def generate_f0s(self):
        print('generating f0')
        with tqdm.tqdm(self.filtered_dict.values()) as pbar:
            for entry in pbar:
                if not entry['note_str'] in self.f0s:
                    audio_path = os.path.join(self.raw_dir, entry['note_str']+'.wav')
                    audio, _sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.length)
                    f0 = compute_f0(audio, sample_rate=self.sample_rate, frame_rate=75, viterbi=True)
                    self.f0s[entry['note_str']] = f0
        with open(self.f0_file_path, 'wb') as f:
            pickle.dump(self.f0s, f)
    
    def create_splits(self, splits, shuffle_files):
        nb_files = len(self.filtered_keys)
        if (shuffle_files):
            idx = np.random.permutation(nb_files)
        else:
            idx = np.linspace(0, nb_files-1, nb_files).astype('int')
        self.train_idx = idx[:int(splits[0]*nb_files)]
        self.valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
        self.test_idx = idx[int((splits[0]+splits[1])*nb_files):]

    def switch_set(self, name):
        if name == 'train':
            self.curr_idx = self.train_idx
        if name == 'valid':
            self.curr_idx = self.valid_idx
        if name == 'test':
            self.curr_idx = self.test_idx
        return self

    def __getitem__(self, idx):
        output = {}
        index = self.curr_idx[idx]
        note = self.filtered_dict[self.filtered_keys[index]]
        file_name = os.path.join(self.raw_dir, note['note_str']+'.wav')
        output['audio'], _sr = librosa.load(file_name, sr=self.sample_rate, duration=self.length)
        # output['audio'] = torchaudio.load_wav(file_name)[0][0]
        # _sr, audio = readwav(file_name)
        # output['audio'] = torch.from_numpy(audio.astype(np.float32))
        # instrument label
        inst_fam = self.insts_dict[note['instrument_family_str']]
        inst_onehot = torch.eye(len(self.use_insts))[int(inst_fam)]
        # Quality label
        quality_vec = torch.Tensor(note['qualities'])
        if self.use_quality:
            output['attributes'] = torch.cat([inst_onehot, quality_vec])
        else:
            output['attributes'] = inst_onehot
        output['instrument'] = inst_onehot
        output['quality'] = quality_vec
        # f0
        f0 = torch.from_numpy(self.f0s[note['note_str']]).float()
        output['f0_hz'] = f0
        return output

    def __len__(self):
        return len(self.curr_idx)

class TinySOLDataset(Dataset):
    """
    Just retrieve audio from a folder with bunch of raw audio (wav) 
    """
    def __init__(self, base_dir, sample_rate=16000, length=4.0):
        # TODO: Use splits from csv
        self.base_dir = base_dir
        # Retrieve list of files
        self.csv_file = os.path.join(self.base_dir, 'TinySOL_metadata.csv')
        self.audio_dir = os.path.join(self.base_dir, 'TinySOL2020')
        with open(self.csv_file, newline='') as f:
            reader = csv.DictReader(f)
            self.csv_data = list(reader)
        self.sample_rate = sample_rate
        self.length = length
        self.total_samples = self.length * sample_rate
        # Analyze dataset
        self.analyze_dataset()
        # Create splits
        # self.create_splits(splits, shuffle_files)
        self.split_folds()
        self.switch_set('train')
        self.insts = ['Bass Tuba', 'French Horn', 'Trombone', 'Trumpet in C', 'Accordion', 'Cello', 'Contrabass', 'Viola', 'Violin', 'Alto Saxophone', 'Bassoon', 'Clarinet in Bb',  'Flute', 'Oboe']
        self.insts_dict = {inst: k for k, inst in enumerate(self.insts)}
        self.n_frames = 0

    def analyze_dataset(self):
        # create f0 or not
        # check number of resampled_files files
        resampled_files = glob.glob('{0}/**/*_{1}.npy'.format(self.audio_dir, self.sample_rate), recursive=True)
        if len(resampled_files) < len(self.csv_data):
            self.generate_f0s()

    def load_audio(self, file_path):
        audio, _sr = librosa.load(file_path, sr=self.sample_rate, duration=self.length)
        n_samples = audio.shape[0]
        # load fixed length audio by zero-padding
        audio = np.pad(audio, (0, int(self.total_samples-n_samples)), 'constant')
        # audio = torch.from_numpy(audio).float()
        # audio = F.pad(audio, (0, int(self.total_samples-n_samples)), 'constant', 0)
        return audio

    def generate_f0s(self):
        audio_paths = []
        f0_paths = []
        print('generating f0')
        with tqdm.tqdm(self.csv_data) as pbar:
            for entry in pbar:
                audio_path = os.path.join(self.audio_dir, entry['Path'])
                f0_path = os.path.splitext(entry['Path'])[0]+'_f0.npy'
                f0_path = os.path.join(self.audio_dir, f0_path)
                # save audio that has been cropped and resampled 
                resample_path = os.path.splitext(entry['Path'])[0] + '_{0}.npy'.format(self.sample_rate)
                resample_path = os.path.join(self.audio_dir, resample_path)
                if not os.path.exists(resample_path):
                    audio = self.load_audio(audio_path)
                    np.save(resample_path, audio)
                    if not os.path.exists(f0_path):
                        f0 = compute_f0(audio, sample_rate=self.sample_rate, frame_rate=50, viterbi=True)
                        np.save(f0_path, f0)

    # def create_splits(self, splits, shuffle_files):
    #     nb_files = len(self.csv_data)
    #     if (shuffle_files):
    #         idx = np.random.permutation(nb_files)
    #     else:
    #         idx = np.linspace(0, nb_files-1, nb_files).astype('int')
    #     self.train_idx = idx[:int(splits[0]*nb_files)]
    #     self.valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
    #     self.test_idx = idx[int((splits[0]+splits[1])*nb_files):]

    def split_folds(self, train_idx=[0,1,2], valid_idx=[3], test_idx=[4]):
        self.folds = [ [], [], [], [], [] ]
        for entry in self.csv_data:
            # look up which fold an entry is in
            self.folds[int(entry['Fold'])].append(entry)
        self.train_fold = []
        self.valid_fold = []
        self.test_fold = []
        for i in train_idx:
            self.train_fold.extend(self.folds[i])
        for i in valid_idx:
            self.valid_fold.extend(self.folds[i])
        for i in test_idx:
            self.test_fold.extend(self.folds[i])


    def switch_set(self, name):
        if name == 'train':
            self.curr_data = self.train_fold
        if name == 'valid':
            self.curr_data = self.valid_fold
        if name == 'test':
            self.curr_data = self.test_fold
        return self

    def __getitem__(self, idx):
        output = {}
        data = self.curr_data[idx]
        #load instrument info
        file_path = os.path.join(self.audio_dir, data['Path'])
        inst_fam = data['Instrument (in full)']
        inst_onehot = torch.eye(len(self.insts))[self.insts_dict[inst_fam]]
        output['attributes'] = inst_onehot
        #load audio
        resample_path = os.path.splitext(data['Path'])[0] + '_{0}.npy'.format(self.sample_rate)
        resample_path = os.path.join(self.audio_dir, resample_path)
        audio = np.load(resample_path)
        output['audio'] = torch.from_numpy(audio).float()
        # load f0 file
        f0_path = os.path.splitext(data['Path'])[0]+'_f0.npy'
        f0_path = os.path.join(self.audio_dir, f0_path)
        f0 = np.load(f0_path)
        output['f0_hz'] = torch.from_numpy(f0).float()

        return output

    def __len__(self):
        return len(self.curr_data)

def load_dataset(data_path, feature_settings):
    dset_train = SoundDataset(data_path, feature_settings).switch_set('train')
    dset_valid = copy.deepcopy(dset_train).switch_set('valid')
    dset_test = copy.deepcopy(dset_train).switch_set('test')
    return dset_train, dset_valid, dset_test

def load_raw_dataset(data_path, sr=16000):
    dset_train = AudioDataset(data_path, sample_rate=sr).switch_set('train')
    dset_valid = copy.deepcopy(dset_train).switch_set('valid')
    dset_test = copy.deepcopy(dset_train).switch_set('test')
    return dset_train, dset_valid, dset_test

def load_nsynth_dataset(data_path, sr=16000):
    dset_train = NsynthDataset(data_path, sample_rate=sr).switch_set('train')
    dset_valid = copy.deepcopy(dset_train).switch_set('valid')
    dset_test = copy.deepcopy(dset_train).switch_set('test')
    return dset_train, dset_valid, dset_test

def load_tinysol_dataset(data_path, sr=16000):
    dset_train = TinySOLDataset(data_path, sample_rate=sr).switch_set('train')
    dset_valid = copy.deepcopy(dset_train).switch_set('valid')
    dset_test = copy.deepcopy(dset_train).switch_set('test')
    return dset_train, dset_valid, dset_test

def load_filtnsynth_dataset(data_path, filt_setting='a', use_quality=False, sr=16000):
    filter_args = {}
    if filt_setting == 'a':
        filter_args['use_insts'] = ['brass', 'flute', 'guitar', 'keyboard', 'mallet', 'reed', 'string', 'vocal']
        filter_args['ng_source_list'] = ['electronic', 'synthetic']
        filter_args['ng_quality_list'] = ['percussive']
        filter_args['pitch_lower'] = 48
        filter_args['pitch_upper'] = 72
    elif filt_setting == 'b':
        filter_args['use_insts'] = ['brass', 'flute', 'guitar', 'keyboard', 'mallet', 'reed']
        filter_args['ng_source_list'] = ['electronic', 'synthetic']
        filter_args['ng_quality_list'] = ['percussive', 'reverb']
        filter_args['pitch_lower'] = 48
        filter_args['pitch_upper'] = 72
    elif filt_setting == 'c':
        filter_args['use_insts'] = ['brass', 'flute', 'guitar', 'keyboard', 'reed']
        filter_args['ng_source_list'] = ['electronic', 'synthetic']
        filter_args['ng_quality_list'] = ['percussive', 'reverb']
        filter_args['pitch_lower'] = 36
        filter_args['pitch_upper'] = 84
    elif filt_setting == 'd':
        filter_args['use_insts'] = ['brass', 'flute', 'guitar', 'keyboard', 'mallet', 'reed']
        filter_args['ng_source_list'] = ['electronic', 'synthetic']
        filter_args['ng_quality_list'] = ['percussive', 'reverb']
        filter_args['pitch_lower'] = 36
        filter_args['pitch_upper'] = 84
    elif filt_setting == 'e':
        filter_args['use_insts'] = ['brass', 'flute', 'guitar', 'mallet', 'reed', 'string']
        filter_args['ng_source_list'] = ['electronic', 'synthetic']
        filter_args['ng_quality_list'] = []
        filter_args['pitch_lower'] = 48
        filter_args['pitch_upper'] = 72
    dset_train = FilteredNsynthDataset(data_path, filter_args, sample_rate=sr).switch_set('train')
    dset_valid = copy.deepcopy(dset_train).switch_set('valid')
    dset_test = copy.deepcopy(dset_train).switch_set('test')
    return dset_train, dset_valid, dset_test

if __name__ == "__main__":
    dt, sv, dtest = load_tinysol_dataset('Datasets/TinySOL/')
    print(dt[0]['f0'])
    