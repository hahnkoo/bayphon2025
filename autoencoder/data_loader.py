"""Data loader
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import glob, argparse, sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def pad_to_longest(batch):
	"""Pad samples so every sample is as long as the longest one."""
	max_len = 0
	for sample in batch:
		L, D = sample.shape
		if max_len < L: max_len = L
	out = []
	for sample in batch:
		L, D = sample.shape
		entry = nn.functional.pad(sample, (0, 0, 0, max_len - L))
		out.append(entry)
	return torch.stack(out)

def trim_to_shortest(batch):
	"""Trim samples so every sample is as long as the shortest one."""
	min_len = 1e+10 
	for sample in batch:
		L, D = sample.shape
		if min_len > L: min_len = L
	out = []
	for sample in batch:
		L, D = sample.shape
		start = 0
		if L > min_len:
			start = torch.randint(L - min_len, (1,)).item()
		entry = sample[start:start+min_len]
		out.append(entry)
	return torch.stack(out)

def pad_scatter(small, target):
	"""Add entries in small so its length goes up to target."""
	remainder = target - len(small)
	if remainder == 0:
		out = small
	else:
		every = len(small) // remainder
		n_to_insert = len(small) // every - 1
		expected_length = len(small) + n_to_insert
		if expected_length > target: n_to_insert -= (expected_length - target) 
		out = []
		n_inserted = 0
		for i in range(len(small)):
			row = small[i]
			out.append(row)
			if i % every == 0 and i != 0 and n_inserted < n_to_insert:
				out.append(row)
				n_inserted += 1
		out = torch.vstack(out)
		still_left = target - len(out)
		pad_top = still_left // 2
		pad_bottom = still_left - pad_top
		out = torch.from_numpy(np.pad(out, ((pad_top, pad_bottom), (0, 0)), mode='edge'))
	return out

def concatenate(x1, x2):
	"""Concatenate two arrays of possibly different lengths."""
	n1 = x1.shape[0]; n2 = x2.shape[0]
	if n1 > n2: x_big = x1; x_small = x2
	else: x_big = x2; x_small = x1
	n_big = x_big.shape[0]; n_small = x_small.shape[0]
	scale_factor = n_big // n_small
	x_small = torch.repeat_interleave(x_small, scale_factor, dim=0)
	x_small = pad_scatter(x_small, n_big) 
	return torch.concatenate((x_big, x_small), axis=1)

def combine_features(csv_dirs, csv_name, normalize, eps=1e-20, log=False):
	"""Combine features from csvs across different directories."""
	try:
		x = load_features(csv_dirs[0] + '/' + csv_name, normalize, eps=eps, log=log)
		if len(x.shape) == 1: x = x.reshape(1, -1)
		for i in range(1, len(csv_dirs)):
			y = load_features(csv_dirs[i] + '/' + csv_name, normalize, eps=eps, log=log)
			if len(y.shape) == 1: y = y.reshape(1, -1)
			x = concatenate(x, y)
		return x
	except: return None

def normalize_features_old(x, eps=1e-20):
	mx = np.max(x) + eps
	return x / mx

def normalize_features(x, eps=1e-20, scale=10, log=False):
	mn = np.min(x)
	mx = np.max(x) + eps
	n = (x - mn) / (mx - mn) * scale
	if log: n = np.log(n + eps)
	return n

def load_features(csv, normalize, eps=1e-20, log=False):
	try:
		x = pd.read_csv(csv, header=None).to_numpy()
		if normalize: x = normalize_features(x, eps=eps, log=log)
		return torch.from_numpy(x).float()
	except:
		return torch.zeros(1, 1)

def load_csv_list(csv_list):
	with open(csv_list) as f:
		out = [line.strip() for line in f]
	return out


class TrainData(Dataset):

	def __init__(self, csv_dirs, csv_list, normalize, eps=1e-20, log=False):
		self.load_frames(csv_dirs, csv_list, normalize, eps=eps, log=log)
		
	def __len__(self):
		return len(self.frames)
	
	def __getitem__(self, idx):
		return self.frames[idx]

	def load_frames(self, csv_dirs, csv_list, normalize, eps=1e-20, log=False):
		self.frames = []
		for csv in csv_list:
			x = combine_features(csv_dirs, csv, normalize, eps=eps, log=log)
			if x is None: pass
			else: self.frames.append(x)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_dirs', type=str)
	parser.add_argument('--csv_list', type=str)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--normalize', action='store_true')
	parser.add_argument('--log', action='store_true')
	args = parser.parse_args()
	csv_dirs = args.csv_dirs.split(',')
	csv_list = load_csv_list(args.csv_list)
	data = TrainData(csv_dirs, csv_list, args.normalize, log=args.log)
	loader = DataLoader(data, batch_size=args.batch_size, collate_fn=trim_to_shortest)
	for x in loader: print(x.shape) 
