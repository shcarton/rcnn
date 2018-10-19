import word2vec


# default_infile = 'data/iac_fourforums_posts.txt'
# default_pfile = 'data/iac_fourforums_phrases.txt'
# default_outfile = 'embeddings/iac_fourforums_300.bin'

default_infile = 'data/wiki/wiki_attack_text.txt'
default_pfile = 'data/wiki/wiki_attack_phrases.txt'
default_outfile = 'embeddings/wiki_attack_300.bin'

default_nd = 300

threads=8

def main():
	print 'Running word2vec on {}, dumping to {}. {} hidden dimensions. Using {} as an intermediate file.'.format(default_infile, default_outfile, default_nd, default_pfile)

	print 'Creating intermediate file.'
	word2vec.word2phrase(default_infile, default_pfile, verbose=True)

	print 'Running word2vec algorithm'
	word2vec.word2vec(default_pfile, default_outfile, size=default_nd,verbose=True, threads=8)
	print 'Done'

if __name__ == '__main__':
	main()