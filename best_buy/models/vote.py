from collections import defaultdict
import operator

def merge_answers(xtrees_preds, forest_preds, tf_preds, knn_preds):
	out = []
	for i in range(len(forest_preds)):
		scores = {'forest':forest_preds[i], 'tf':tf_preds[i], 'xtrees':xtrees_preds[i], 'knn':knn_preds[i]}
		ranked = vote(scores)
		out.append(ranked)
	return out

""" Return an array of hashes:
[ {'a':0.5, 'b':0.2}, {'a':1.0} ] """
def vote(predictions):
	weights = {'tf':0.64, 'forest':0.62, 'xtrees':0.63, 'knn':0.54}
	combined = defaultdict(int)
	for name,p in predictions.items():
		weight = weights[name]
		for index,pred in enumerate(p):
			score = weight/(index+1)
			combined[pred] += score

	# Sort and return the top five.
	ranked = sorted(combined.iteritems(), key=operator.itemgetter(1))
	ranked.reverse()
	best = []
	for r in ranked:
		if len(best) < 5:
			best.append(r[0])
	#print predictions
	#print combined
	#print best
	#print "\n"
	return best
