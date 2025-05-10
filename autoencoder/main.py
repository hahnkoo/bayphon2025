"""Main script for autoencoder segmentation
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import re, sys, glob, argparse, time, copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import signal, spatial
import data_loader

def load_timit(phn):
	"""Get boundaries from a TIMIT PHN file.
	- This includes beginning and end of the recording.
	"""
	out = [0.0]
	with open(phn) as f:
		for line in f:
			b, e, p = line.strip().split()
			e = float(e) / 16000
			out.append(e)
	return out

def load_config(config_file):
	out = {}
	if not config_file is None:
		with open(config_file) as f:
			for line in f:
				line = re.sub('#.+', '', line).strip()
				if line != '':
					ll = line.split(',')
					out[ll[0].strip()] = ll[1].strip()
	return out

def iterate(model, x, loss_function, optimizer):
	optimizer.zero_grad()
	y = x.clone()
	y_ = model(x)
	if str(loss_function) == 'CosineEmbeddingLoss()':
		y_ = y_.reshape(-1, y_.shape[-1])
		y = y.reshape(-1, y.shape[-1])
		loss = loss_function(y_, y, torch.ones(y_.shape[0]))
	else: loss = loss_function(y_, y)
	loss.backward()
	optimizer.step()
	return loss.item()

def train(model, train_loader, n_epochs, learning_rate, save=None, min_delta=0.0001, patience=5, burn_in=10):
	st = time.time() 
	loss_function = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	best = torch.inf
	best_params = copy.deepcopy(model.state_dict())
	n_bad_epochs = 0
	loss = 0 
	for n in range(n_epochs):
		loss = 0
		for x in train_loader:
			batch_loss = iterate(model, x, loss_function, optimizer)
			sys.stderr.write('# Epoch ' + str(n+1) + ' batch loss = ' + str(batch_loss) + '\r')
			loss += batch_loss
		sys.stderr.write('\n# Loss after epoch ' + str(n+1) + ' = ' + str(loss) + '\n')
		if loss < best - min_delta:
			best = loss
			best_params = copy.deepcopy(model.state_dict())
			n_bad_epochs = 0
			if save: torch.save(best_params, save)
		elif n >= burn_in:
			n_bad_epochs += 1
		if n_bad_epochs > patience: break
	model.load_state_dict(best_params)
	et = time.time() 
	sys.stderr.write('# Training complete in ' + str(et-st) + ' seconds.\n')
	return loss

def sum_squared_diff(x):
	"""Calculate sum of squared differences between two adjacent rows:
	sum_i (x[t, i] - x[t-1, i])**2
	"""
	xp = np.pad(x, ((1,0),(0,0)), 'edge')
	d = xp[1:, :] - xp[:-1, :]
	return (d*d).sum(axis=1)

def normalize(x):
	mx = np.max(x) + 1e-20
	mn = np.min(x)
	return (x - mn) / (mx - mn)

def segment(representation, height, threshold):
	d = normalize(sum_squared_diff(representation))
	bs, _ = signal.find_peaks(d, height=height, threshold=threshold)
	return d, bs 

def time_stamp(boundaries, frame_shift, silent_interval_file=None):
	stamps = boundaries * frame_shift
	if silent_interval_file:
		sil = load_silent_intervals(silent_interval_file)
		out = []
		for t in stamps:
			in_silence = False
			for s, e in sil:
				if t >= s and t <= e:
					in_silence = True
					break
			if not in_silence: out.append(t)
		out = np.array(out)
		return out
	else:
		return stamps

def plot(model, x, height, threshold, phn, frame_shift):
	y = x.clone().squeeze(0).detach().numpy()
	y_ = model.forward(x).squeeze(0).detach().numpy()
	h = model.represent(x).squeeze(0).detach().numpy()
	ad, hbs = segment(h, height, threshold)
	sys.path.append('../')
	rbs = load_timit(phn)[1:-1]
	t = np.arange(y.shape[0]) * frame_shift

	fig, ax = plt.subplots(3, 1, figsize=(24, 8))
	ax[0].pcolormesh(t, np.arange(y.shape[1]), y.T)
	for b in rbs: ax[0].axvline(b, color='red', linestyle='dashed')
	ax[0].set_title('(a) original log mel spectrogram with reference boundaries')
	ax[0].set_xticks([]); ax[0].set_yticks([])

	ax[1].pcolormesh(y_.T)
	for b in hbs: ax[1].axvline(b, color='red', linestyle='dashed')
	ax[1].set_title('(b) reconstructed log mel spectrogram with hypothesized boundaries')
	ax[1].set_xticks([]); ax[1].set_yticks([])

	ax[2].pcolormesh(h.T)
	ax[2].plot(normalize(ad) * h.shape[1], color='red')
	ax[2].set_xlim(left=0, right=len(ad))
	ax[2].set_title('(c) learned representation with difference between adjacent frames')
	ax[2].set_xticks([]); ax[2].set_yticks([])

	plt.subplots_adjust(hspace=0.2)
	plt.savefig('./out.png', bbox_inches='tight')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--module', type=str, default='transformer')
	parser.add_argument('--config', type=str, default='transformer.config')
	parser.add_argument('--n_features', type=int, default=80)
	parser.add_argument('--normalize_feature', action='store_true')
	parser.add_argument('--save', type=str)
	parser.add_argument('--load', type=str)
	parser.add_argument('--train_csv_list', type=str)
	parser.add_argument('--train_csv_dirs', type=str)
	parser.add_argument('--test_csv_list', type=str)
	parser.add_argument('--test_csv_dirs', type=str)
	parser.add_argument('--plot', action='store_true')
	parser.add_argument('--plot_phn', type=str)
	parser.add_argument('--outdir', type=str)
	parser.add_argument('--n_epochs', type=int, default=10)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--min_delta', type=float, default=0.0001)
	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument('--burn_in', type=int, default=10)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--frame_shift', type=float, default=0.01)
	parser.add_argument('--height', type=float, default=0.05)
	parser.add_argument('--threshold', type=float, default=0.0)
	parser.add_argument('--finetune', type=str, default='1000,0.01')
	args = parser.parse_args()
	config = load_config(args.config)
	config['input_dim'] = args.n_features
	module = __import__(args.module)
	torch.manual_seed(123)
	m = module.Model(config)
	for name, param in m.named_parameters():
		if param.requires_grad:
			if 'bias' in name: nn.init.constant_(param, 0)
			elif 'weight' in name:
				try: nn.init.xavier_uniform_(param)
				except: nn.init.uniform_(param)
	if args.load:
		m.load_state_dict(torch.load(args.load))
		sys.stderr.write('# Model parameters loaded from ' + args.load + '\n')
	if args.train_csv_list and args.train_csv_dirs:
		train_csv_dirs = args.train_csv_dirs.split(',')
		train_csv_list = data_loader.load_csv_list(args.train_csv_list)
		train_data = data_loader.TrainData(train_csv_dirs, train_csv_list, args.normalize_feature)
		train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=data_loader.trim_to_shortest)
		train(m, train_loader, args.n_epochs, args.learning_rate, save=args.save, min_delta=args.min_delta, patience=args.patience)
	if args.test_csv_list and args.test_csv_dirs:
		test_csv_dirs = args.test_csv_dirs.split(',')
		test_csv_list = data_loader.load_csv_list(args.test_csv_list)
		for csv in test_csv_list:
			m_csv = copy.deepcopy(m)
			if args.finetune:
				finetune_epochs, finetune_lr = args.finetune.split(',')
				finetune_epochs = int(finetune_epochs)
				finetune_lr = float(finetune_lr)
				sample = data_loader.TrainData(test_csv_dirs, [csv], args.normalize_feature)
				sample_loader = DataLoader(sample, batch_size=1)
				train(m_csv, sample_loader, finetune_epochs, finetune_lr, min_delta=args.min_delta, patience=args.patience)
			x = data_loader.combine_features(test_csv_dirs, csv, args.normalize_feature).unsqueeze(0)
			if args.plot:
				plot(m_csv, x, args.height, args.threshold, args.plot_phn, args.frame_shift) 
			if args.outdir:
				h = m_csv.represent(x)
				print(h.shape)
				representation = h.squeeze(0).detach().numpy()
				d, bs = segment(representation, args.height, args.threshold)
				bts = time_stamp(bs, args.frame_shift)
				handle = csv.split('/')[-1].split('.')[0]
				ofn = args.outdir + '/' + handle + '.BND'
				with open(ofn, 'w') as f:
					for bt in bts: f.write(str(bt) + '\n')
