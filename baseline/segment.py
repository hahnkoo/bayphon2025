"""A baseline model for unsupervised phoneme segmentation
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import sys, argparse, glob, re, copy
import numpy as np
from scipy import signal

def cosine_similarity(x, y):
	dp = np.sum(x * y, axis=1)
	return dp / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

def delta(x, n):
	"""
	Average n frames to the left and n frames to the right for each frame.
	Calculate the cosine distance between the two averages.
	"""
	s = x.copy()
	for w in range(1, n):
		xp = np.pad(x, ((w, 0), (0, 0)), 'edge')
		s += xp[w:, :]
	s /= n
	sl = np.pad(s, ((n, 0), (0, 0)), 'edge')
	sr = np.pad(s, ((0, n), (0, 0)), 'edge')
	d = 1 - cosine_similarity(sr[n:, :], sl[:-n, :])
	return d 
	 
def normalize_vals(x):
	"""Normalize values so they're in [0, 1]."""
	return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_boundaries(cs, height, threshold, width, normalize):
	"""Get boundaries from cs."""
	d = delta(cs, width)
	if normalize: d = normalize_vals(d)
	bs, _ = signal.find_peaks(d, height=height, threshold=threshold)
	return d, bs


def main(args):
	for csv in glob.glob(args.csv_dir + '/*.csv'):
		handle = csv.split('/')[-1].split('.')[0]
		sys.stderr.write('# Processing ' + handle + '\r')
		vecs = np.genfromtxt(csv, delimiter=',')
		ofn = args.outdir + '/' + handle + '.BND'
		avg, bs = get_boundaries(vecs, args.height, args.threshold, args.delta_width, args.normalize)
		with open(ofn, 'w') as of: 
			for b in bs:
				b_sec = b * args.frame_shift
				of.write(str(b_sec) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_dir', type=str)
	parser.add_argument('--outdir', type=str)
	parser.add_argument('--sampling_rate', type=int, default=16000)
	parser.add_argument('--frame_shift', type=float, default=0.01)
	parser.add_argument('--delta_width', type=int, default=2)
	parser.add_argument('--height', type=float, default=0.05)
	parser.add_argument('--threshold', type=float, default=0.0)
	parser.add_argument('--normalize', action='store_true')
	args = parser.parse_args()
	main(args)
