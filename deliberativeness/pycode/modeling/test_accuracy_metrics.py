import sklearn.metrics as mt
import warnings
warnings.filterwarnings("ignore")
import numpy as np

def main():
	r1 = [0,0,0,0,0,0]
	r2 = [1,1,1,1,1,1]
	r3 = [0,1,1,1,1,1]
	r4 = [1,0,0,0,0,0]
	r5 = [0,0,0,0,0,1]
	r6 = [0,0,0,0,1,1]

	pairs = [
		[r1,r1],
		[r1,r5],
		[r5,r1],
		[r6,r1],
		[r2,r3],
		[r3,r2]
	]

	for true_rationale, predicted_rationale in pairs:
		evaluate_rationale(true_rationale, predicted_rationale)

def evaluate_rationale(true_rationale, predicted_rationale):
	print 'True rationale:      {}'.format(true_rationale)
	print 'Predicted rationale: {}'.format(predicted_rationale)

	print '\tAccuracy: {:.3f}'.format(mt.accuracy_score(true_rationale, predicted_rationale))
	print '\tF1: {:.3f}'.format(mt.f1_score(true_rationale, predicted_rationale))
	print '\tF1 micro: {:.3f}'.format(mt.f1_score(true_rationale, predicted_rationale,average='micro'))
	print '\tF1 macro: {:.3f}'.format(mt.f1_score(true_rationale, predicted_rationale,average='macro'))
	print '\tF1 weighted: {:.3f}'.format(mt.f1_score(true_rationale, predicted_rationale,average='weighted'))

	print '\tPrecision: {:.3f}'.format(mt.precision_score(true_rationale, predicted_rationale) if np.any(predicted_rationale) else np.NaN)
	print '\tRecall: {:.3f}'.format(mt.recall_score(true_rationale, predicted_rationale) if np.any(true_rationale) else np.NaN)




if __name__ == "__main__":
	main()