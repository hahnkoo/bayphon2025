"""Transformer autoencoder
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import sys, glob, argparse
import numpy as np
import pandas as pd
import torch
from torch import nn


def sinusoidal_positional_encoding(x):
	"""Sinusoidal positional encoding for x:
	pos_{i, 2j} = \sin ( \frac{i}{10000^{2j/d}} )
	pos_{i, 2j+1} = \cos ( \frac{i}{10000^{2j/d}} )
	"""
	b, l, d = x.shape
	n = d // 2 + (d % 2 == 1)
	pos_i = torch.arange(l).view(-1, 1).repeat(1, n)
	pos_j = 10000 ** (torch.arange(n) / d)
	pos_even = torch.sin(pos_i / pos_j)
	pos_odd = torch.cos(pos_i / pos_j)
	pos = torch.stack([pos_even, pos_odd], dim=-1).view(l, -1)
	pos = pos[:, :d]
	pos = pos.repeat(b, 1, 1)
	return pos


class Conv(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.filter = nn.Sequential(
						nn.Conv1d(in_channels, out_channels, kernel_size=1),
						nn.ReLU(),
						nn.BatchNorm1d(out_channels)
					)

	def forward(self, x):
		return self.filter(x)


class MaskedAttention(nn.Module):

	def __init__(self, input_dim, n_heads, mask_window_size):
		super().__init__()
		self.attention = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
		self.attn_output_weights = None
		self.mask_window_size = mask_window_size
		self.mask = torch.ones(1, 1).bool()

	def gen_mask(self, x):
		L = x.shape[1]
		if self.mask.shape[-1] != L:
			self.mask = torch.eye(L, L)
			if self.mask_window_size > 0:
				self.mask = torch.ones(L, L)
				for n in range(L):
					left_edge = max(0, n - self.mask_window_size)
					for i in range(left_edge, n):
						self.mask[n, i] = 0
					right_edge = min(n + 1 + self.mask_window_size, L)
					for i in range(n+1, right_edge):
						self.mask[n, i] = 0
			self.mask = self.mask.bool()

	def forward(self, x):
		self.gen_mask(x)
		attn_output, attn_output_weights = self.attention(x, x, x, attn_mask=self.mask)
		self.attn_output_weights = attn_output_weights 
		return attn_output 


class EncoderLayer(nn.Module):

	def __init__(self, input_dim, n_heads, dim_feedforward, mask_window_size):
		super().__init__()
		self.attention = MaskedAttention(input_dim, n_heads, mask_window_size)
		self.feedforward = nn.Sequential(nn.Linear(input_dim, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, input_dim))

	def forward(self, x):
		h = self.attention(x)
		y_ = self.feedforward(h)
		return y_


class Model(nn.Module):

	def __init__(self, config):
		super().__init__()
		input_dim = int(config.get('input_dim', 80))
		embedding_dim = int(config.get('embedding_dim', 128))
		n_heads = int(config.get('n_heads', 16))
		dim_feedforward = int(config.get('dim_feedforward', 128))
		mask_window_size = int(config.get('mask_window_size', 0))
		self.embed = Conv(input_dim, embedding_dim) 
		self.enc = EncoderLayer(embedding_dim, n_heads, dim_feedforward, mask_window_size)
		self.fc = nn.Linear(embedding_dim, input_dim)

	def represent(self, x):
		x = x.transpose(1, 2)
		h = self.embed(x)
		h = h.transpose(1, 2)
		h += sinusoidal_positional_encoding(h)
		h = self.enc(h)
		return h
		
	def forward(self, x):
		h = self.represent(x)
		y_ = self.fc(h)
		return y_

