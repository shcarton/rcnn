import word2vec
import gzip

# default_infile = 'embedding/iac_fourforums_300.bin'
# default_outfile = 'embedding/iac_fourforums_300.txt.gz'


default_infile = 'embeddings/wiki_attack_300.bin'
default_outfile = 'embeddings/wiki_attack_300.txt.gz'

def main():
	print 'Converting word2vec binary file at {}, to gzipped text file at {}'.format(default_infile, default_outfile)

	print 'Loading vector file'
	model = word2vec.load(default_infile)

	print 'Dumping file'
	with gzip.open(default_outfile, 'wb') as of:
		for i, v in enumerate(model.vocab):
			of.write(v.encode('utf-8'))
			of.write(' ')
			of.write(' '.join([str(x) for x in model.vectors[i]]))
			of.write('\n')
	print 'Done'

if __name__ == '__main__':
	main()