# import matplotlib
# matplotlib.use('Agg')

import pandas as pd
# import re
# import nltk
import csv
# from datetime import datetime

from putil import *
import os

'''
This script takes in TSV files available at e.g. https://figshare.com/articles/Wikipedia_Talk_Labels_Aggression/4267550
and creates data csvs of them

In particular it takes an annotation file, which is crowdflower use responses, and an annotated_comment file, which is
comment text. It performs a join, aggregates the combined score for each comment, and
'''

# #Comments file
# ac_file = '../../data/raw/wiki/toxicity/toxicity_annotated_comments.tsv'
# #Annotation file
# a_file = '../../data/raw/wiki/toxicity/toxicity_annotations.tsv'
# outdir = '../../data/processed/wiki/toxicity'
# outprefix = 'wiki_toxicity'
# outsuffix = '.csv'
# vcol = 'toxicity'
# # binarize_function = lambda y: int(y > 0) #Toxicity scores are in [-2,-1,0,1,2], but I am really just interested in toxic vs nontoxic


#Comments file
ac_file = '../../data/raw/wiki/personal_attacks/attack_annotated_comments.tsv'
#Annotation file
a_file = '../../data/raw/wiki/personal_attacks/attack_annotations.tsv'
outdir = '../../data/processed/wiki/personal_attacks'
outprefix = 'wiki_attack'
outsuffix = '.csv'
vcol = 'attack'
binarize_function = None


# #Comments file
# ac_file = '../../data/raw/wiki/aggression/aggression_annotated_comments.tsv'
# #Annotation file
# a_file = '../../data/raw/wiki/aggression/aggression_annotations.tsv'
# outdir = '../../data/processed/wiki/aggression'
# outprefix = 'wiki_aggression'
# outsuffix = '.csv'
# vcol = 'aggression'


gcol = 'rev_id'
tcol = 'comment'

platform_id = 0 #wikipedia platform ID is 0



pd.options.display.width = 200
def main():
	print 'Aggregating Wikipedia annotation files {} and {} to training development and test csvs in '.format(ac_file, a_file, outdir)
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	acdf = pd.read_csv(ac_file, delimiter='\t')
	adf = pd.read_csv(a_file,delimiter='\t')

	print '{} items loaded. {} Turker responses loaded.'.format(acdf.shape[0], adf.shape[0])

	# if binarize_function:
	# 	print 'Binarizing all target values'
	# 	adf[vcol] = adf[vcol].apply(binarize_function)
	# else:
	# 	print 'Not manipulating target values before calculating means'

	gadf = adf.groupby(by=gcol)

	odf = acdf.join(gadf[vcol].mean(),on=gcol,how='inner')
	print 'Inner-joined data frame has {} rows.'.format(odf.shape[0])

	odf.rename(columns={"rev_id":"platform_comment_id", "comment":"original_text", vcol:"target"},inplace=True)
	odf['platform_id'] = platform_id
	odf['url'] = odf['platform_comment_id'].apply(wiki_revid_to_url)
	odf['datetime'] = odf['year'].apply(year2datetime)
	odf['original_text'] = odf['original_text'].apply(replace_wiki_tokens)
	odf[['text','tokenization']] = odf['original_text'].apply(process_text_to_pd)



	#We use the same train/dev/test split as the compilers of the dataset
	tr_odf = odf[odf['split'] == 'train'][labeled_data_columns]
	d_odf = odf[odf['split'] == 'dev'][labeled_data_columns]
	te_odf = odf[odf['split'] == 'test'][labeled_data_columns]
	# a_odf = odf[labeled_data_columns]

	dfs = [tr_odf,d_odf,te_odf]
	sps = ['train','dev','test']

	for i,df in  enumerate(dfs):
		sp = sps[i]
		print '\n{}'.format(sp)

		df = df[(df['text'].notnull()) & (df['text'] != '')]
		print '\n{}'.format(sp)
		of = outdir+'/'+outprefix+'_'+sp+outsuffix



		print 'Writing {} {} rows to {}'.format(df.shape[0], sp, of)
		df.to_csv(of,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')
		analyze_column(df['target'])

	tf = outdir+'/'+outprefix+'_text.txt'
	print 'Writing text to {}'.format(tf)
	# with open(tf,'w') as tff:
	# 	for txt in odf[tcol]:
	# 		tff.write(txt.encode('utf-8'))
	# 		tff.write('\n')
	# tff.close()


	print 'Done'



if __name__ == '__main__':
	main()