"""A script for converting WRD files in TIMIT to lab files.
"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu"

import argparse, glob, re

def wrd2lab(wrd):
	content = ''
	with open(wrd) as f:
		for line in f:
			word = line.strip().split()[-1]
			content += word.upper() + ' '
	content = content.strip()
	dirs = wrd.split('/')
	i = dirs.index('TIMIT')
	filename = '_'.join(dirs[i+1:])
	filename = re.sub('\.WRD', '.lab', filename)
	return filename, content


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--wrd', type=str)
	parser.add_argument('--outdir', type=str)
	args = parser.parse_args()
	for wrd in glob.glob(args.wrd):
		fn, ct = wrd2lab(wrd)
		with open(args.outdir + '/' + fn, 'w') as f:
			f.write(ct + '\n')
