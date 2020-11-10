import torch.optim as optim
from ddspsynth.data import load_nsynth_dataset, load_tinysol_dataset, load_filtnsynth_dataset
from torch.utils.data import DataLoader, Subset
from ddspsynth.classifier import Classifier
from ddspsynth.evaluate import save_losses_csv
import torch
import torch.nn as nn
import os, pickle, argparse, json
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('output',           type=str,                           help='')
parser.add_argument('data_path',        type=str,                           help='')
parser.add_argument('--data_filt',      type=str,   default='a',            help='')
parser.add_argument('--dataset',        type=str,   default='fnsynth',      help='')
# Optimization arguments
parser.add_argument('--batch_size',     type=int,   default=64,             help='')
parser.add_argument('--epochs',         type=int,   default=50,            help='')
parser.add_argument('--lr',             type=float, default=2e-4,           help='')
parser.add_argument('--subset',         type=int,   default=None,           help='')
# performance related arguments
parser.add_argument('--device',         type=str,   default='cuda',         help='Device for CUDA')
parser.add_argument('--nbworkers',      type=int,   default=4,              help='')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.mkdir(args.output)

# set device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Optimization will be on ' + str(args.device) + '.')
# load dataset
suffix = args.data_filt if args.dataset == 'fnsynth' else ''
dataset_file = os.path.join(args.data_path, 'datasets_{0}.pkl'.format(suffix))
if os.path.exists(dataset_file):
    with open(dataset_file, 'rb') as f:
        dset_train, dset_valid, dset_test = pickle.load(f)
else:
    if args.dataset == 'nsynth':
        dset_train, dset_valid, dset_test = load_nsynth_dataset(args.data_path)
    elif args.dataset == 'fnsynth':
        dset_train, dset_valid, dset_test = load_filtnsynth_dataset(args.data_path, args.data_filt)
    elif args.dataset == 'tinysol': 
        dset_train, dset_valid, dset_test = load_tinysol_dataset(args.data_path)
    with open(dataset_file, 'wb') as f:
        pickle.dump([dset_train, dset_valid, dset_test], f)

dl_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.nbworkers, pin_memory=False)
dl_valid = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=False)
dl_test = DataLoader(dset_test, batch_size=args.batch_size, num_workers=args.nbworkers, pin_memory=False)

testbatch = next(iter(dl_test))
n_classes = testbatch['instrument'].shape[1]
model = Classifier(n_classes).to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, threshold=1e-7)

best_acc = 0
stats = torch.zeros(args.epochs, 2)
with tqdm.tqdm(range(args.epochs)) as pbar:
    for i in pbar:
        stats[i, 0] = model.train_epoch(dl_train, loss, optimizer, device)
        stats[i, 1] = model.eval_epoch(dl_valid, device)
        tqdm.tqdm.write('Epoch: {0} Tr. Loss: {1:.5f} Val. Loss: {2:.5f}%'.format(i+1, stats[i, 0], stats[i, 1]*100))
        scheduler.step(stats[i, 1])
        save_losses_csv(stats, os.path.join(args.output, 'losses.csv'))
        if (stats[i, 1] > best_acc):
            # Save model
            best_acc = stats[i, 1]
            torch.save(model.state_dict(), os.path.join(args.output, 'state_dict.pth'))