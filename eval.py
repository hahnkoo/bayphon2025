"""An evaluation script for phoneme segmentation based on Rasanen et al. (2009)

References:
Rasanen, O. J., Laine, U. K., & Altosaar, T. (2009). An improved speech segmentation quality measure: the R-value. Proceedings of Interspeech 2009 (pp. 1851-1854).
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import argparse, glob, re
import numpy as np

def load_hypothesis(fn):
	"""Load hypothesized boundaries.

	Args:
	- fn: path to a file containing boundaries in seconds, one boundary per line; str
	"""
	with open(fn) as f:
		out = [float(line.strip()) for line in f]
	return out


# missed boundary (deletion error): a reference boundary does not have a corresponding hypothesized boundary
# additionally hypothesized boundary (insertion error): presence of a boundary within a segment

def load_timit(phn):
	"""Get boundaries from a TIMIT PHN file.
	- This includes beginning and end of the recording.

	Args:
	- phn: path to a TIMIT .PHN file; str 
	"""
	phones = ['<s>']
	time_stamps = [0.0]
	with open(phn) as f:
		for line in f:
			b, e, p = line.strip().split()
			e = float(e) / 16000
			time_stamps.append(e)
			phones.append(p)
	return time_stamps, phones 

def define_search_regions(boundaries, tolerance_level=0.02):
	"""Define a search region (e.g. +/- 20ms interval) around each boundary.
	- If regions overlap between adjacent boundaries, settle at the mid-point of the overlapping interval.

	Args:
	- boundaries: boundaries in seconds; list of floats
	-- includes 0 and end point of recording 
	- tolerance_level: width of search region on either side of each boundary 
	"""
	t = tolerance_level
	s = boundaries[0]; e = boundaries[-1]
	regions = [[b-t, b+t] for b in boundaries[1:-1]]
	regions[0][0] = max(s, regions[0][0])
	regions[-1][1] = min(e, regions[-1][1])
	for i in range(len(regions)-1):
		if regions[i][1] > regions[i+1][0]:
			mid = (regions[i+1][0] + regions[i][1]) / 2
			regions[i][1] = mid
			regions[i+1][0] = mid
	return regions

def classify(y_, search_regions):
	"""Decide which hypothetical boundaries belong to which search regions.

	Args:
	- y_: hypothesis listing of boundaries in sec; list of floats
	- search_regions: search regions; list of time stamp pairs 
	"""
	b_stranded = y_.copy()
	bi = 0
	sr_covered = {}
	for s, e in search_regions:
		sr_covered[(s, e)] = []
		for i in range(bi, len(y_)):
			if y_[i] >= s and y_[i] < e:
				sr_covered[(s, e)].append(y_[i])
				b_stranded.remove(y_[i]) 
			if y_[i] >= e:
				bi = i
				break
	return sr_covered, b_stranded

def locate_stranded(ref_time_stamps, ref_phones, stranded):
	"""Locate which phone labels the interval containing the stranded hypothesized boundary."""
	rts = np.array(ref_time_stamps)
	last = sum(rts < stranded)
	if last >= len(rts): last = -1
	return ref_phones[last]

def error_analysis(ref_time_stamps, ref_phones, sr_covered, b_stranded):
	"""Error analysis."""
	insertions = []
	deletions = []
	srs = sorted(sr_covered.keys())
	for i in range(len(srs)):
		if len(sr_covered[srs[i]]) == 0:
			deletions.append((ref_phones[i], ref_phones[i+1]))
		for j in range(1, len(sr_covered[srs[i]])):
			insertions.append((ref_phones[i], ref_phones[i+1]))
	for b in b_stranded:
		insertions.append(locate_stranded(ref_time_stamps, ref_phones, b))
	return insertions, deletions
	
def eval_count(y_, phn, tolerance_level=0.02, analyze_errors=False):
	"""Evaluate phoneme segmentation against TIMIT PHN file."""
	ref_time_stamps, ref_phones = load_timit(phn) 
	sr = define_search_regions(ref_time_stamps, tolerance_level=tolerance_level)
	sr_covered, b_stranded = classify(y_, sr)
	n_ref = len(sr_covered) # number of reference boundaries
	n_f = len(y_) # number of hypothesized boundaries
	n_hit = 0 # number of hits
	for x in sr_covered:
		if len(sr_covered[x]) > 0:
			n_hit += 1
	insertions = []; deletions = []
	if analyze_errors:
		insertions, deletions = error_analysis(ref_time_stamps, ref_phones, sr_covered, b_stranded)
	return n_hit, n_ref, n_f, insertions, deletions

def score(n_hit, n_ref, n_f):
	"""Convert eval counts to scores."""
	hr = n_hit / n_ref * 100 # hit-rate
	os = (n_f / n_ref - 1) * 100 # over-segmentation rate
	prc = n_hit / n_f # precision
	rcl = n_hit / n_ref # recall
	f = 2 * prc * rcl / (prc + rcl) # f-score
	r1 = np.sqrt((100 - hr)**2 + os**2)
	r2 = (-os + hr - 100) / np.sqrt(2)
	r = 1 - (abs(r1) + abs(r2)) / 200 # R-value
	return hr, os, prc, rcl, f, r

def summarize_errors(insertions, deletions):
	ins_dict = {}; del_dict = {}
	for pp in insertions:
		if not pp in ins_dict: ins_dict[pp] = 0
		ins_dict[pp] += 1
	for pp in deletions:
		if not pp in del_dict: del_dict[pp] = 0
		del_dict[pp] += 1
	print('# Top 10 insertion errors:')
	ins_items = sorted(ins_dict.items(), key=lambda tuple: tuple[1], reverse=True)
	for pp, cnt in ins_items[:10]: print('##', pp, '\t', cnt)
	print('# Top 10 deletion errors:')
	del_items = sorted(del_dict.items(), key=lambda tuple: tuple[1], reverse=True)
	for pp, cnt in del_items[:10]: print('##', pp, '\t', cnt)

def main(args):
	n_hit_total = 0; n_ref_total = 0; n_f_total = 0
	insertions = []; deletions = []
	for phn in glob.glob(args.phn_dir + '/*.PHN'):
		fid = phn.split('/')[-1]
		hyp = args.hyp_dir + '/' + re.sub('\.PHN$', '.BND', fid)
		y_ = load_hypothesis(hyp)
		n_hit, n_ref, n_f, insertions_i, deletions_i = eval_count(y_, phn, tolerance_level=args.tolerance, analyze_errors=args.error_analysis)
		n_hit_total += n_hit; n_ref_total += n_ref; n_f_total += n_f
		insertions += insertions_i; deletions += deletions_i
	hr, os, prc, rcl, f, r = score(n_hit_total, n_ref_total, n_f_total)
	print('# N_hit =', n_hit_total)
	print('# N_ref =', n_ref_total)
	print('# N_f =', n_f_total)
	print('# hit rate =', hr)
	print('# over segmentation rate =', os)
	print('# precision =', prc)
	print('# recall =', rcl)
	print('# F-score =', f)
	print('# R-value =', r)
	if args.error_analysis: summarize_errors(insertions, deletions)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--hyp_dir', type=str)
	parser.add_argument('--phn_dir', type=str)
	parser.add_argument('--tolerance', type=float, default=0.02)
	parser.add_argument('--error_analysis', action='store_true')
	args = parser.parse_args()
	main(args)
