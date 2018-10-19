import sys
import gzip
import csv

stop_at = -1

def main():
	infile = sys.argv[1]
	outfile = sys.argv[2]
	cols = sys.argv[3].split(',') #comma-delimited list


	print 'Converting gzipped data file at {} into a csv at {}'.format(infile, outfile)
	f = gzip.open(infile)
	lines = []
	n = 0
	for line in f:
		pieces = line.split('\t')
		text = pieces[1].strip()
		numbers = [float(x) for x in pieces[0].split()]
		lines.append(numbers + [text])
		n += 1
		if stop_at > 0 and  n >= stop_at:
			break

	f.close()

	print '{} lines read from gzip file. Now writing to csv.'.format(n)

	with open(outfile, 'wb') as of:
		writer = csv.writer(of, lineterminator='\n', quoting = csv.QUOTE_NONNUMERIC)
		writer.writerow(cols)
		for line in lines:
			writer.writerow(line)


	print 'Done'







if __name__=="__main__":
	main()
