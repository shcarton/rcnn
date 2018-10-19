import sys
import gzip
import csv
import pandas as pd

stop_at = -1
normalize = True


default_infile = 'data/processed/wiki/personal_attacks/wiki_attack_test.csv'
default_outfile = 'data/wiki/wiki_attack_test.txt.gz'

default_columns = ["attack"]
default_text_col = "comment"




def main():
	if len(sys.argv) == 5:
		print 'Reading arguments from command line'
		infile = sys.argv[1]
		outfile = sys.argv[2]
		cols = sys.argv[3].split(',') #comma-delimited list
		text_col = sys.argv[4]
	else:
		print 'Using default arguments'
		infile = default_infile
		outfile = default_outfile
		cols = default_columns
		text_col = default_text_col

	print 'Converting csv data file at {} into a gzipped text file at {}'.format(infile, outfile)

	# df = pd.read_csv(infile,escapechar='\\')
	df = pd.read_csv(infile)
	print '{} lines read from csv file. Now writing to gzip.'.format(df.shape[0])
	# df[text_col] = df[text_col].apply(strip_newlines)



	f = gzip.open(outfile,'wb')
	for i,row in df.iterrows():
		if type(row[text_col]) == str:
			f.write(' '.join(str(x) for x in list(row[cols]))+'\t')
			f.write(row[text_col])
			f.write('\n')

	f.close()



	print 'Done'


def strip_newlines(s):
	return s.replace('\n',' ').replace('\r', ' ').strip()





if __name__=="__main__":
	main()
