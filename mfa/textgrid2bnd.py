"""A script for converting TextGrid file from the Montreal Forced Aligner to files that list phoneme boundaries in seconds. 
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu"

import glob, argparse, re, sys

def stamp_boundaries(textgrid_file):
	with open(textgrid_file) as f:
		begin = False; go = False
		boundaries = []
		for line in f:
			line = line.strip()
			if re.search('name = "phones"', line): begin = True
			if begin and re.search('intervals: size', line): go = True
			if go:
				if re.search('xmax', line):
					t = float(line.split('=')[-1].strip())
					boundaries.append(t)
		boundaries = boundaries[:-1] # last boundary is end of recording, so ignore
		return boundaries

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--TextGrid', type=str)
	parser.add_argument('--outdir', type=str)
	args = parser.parse_args()
	for tg in glob.glob(args.TextGrid):
		ofn = re.sub('\.TextGrid', '.BND', tg.split('/')[-1])
		bs = stamp_boundaries(tg)
		with open(args.outdir + '/' + ofn, 'w') as f:
			for b in bs: f.write(str(b) + '\n')
