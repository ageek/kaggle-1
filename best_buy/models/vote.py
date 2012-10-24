from collections import defaultdict
import operator

def merge_answers(xtrees_preds):#, xtrees2, xtrees3, xtrees4, forest_preds, tf_preds, knn_preds, time_preds):
	out = []
	for i in range(len(forest_preds)):
		scores = {'forest':forest_preds[i], 'tf':tf_preds[i], 'xtrees':xtrees_preds[i], 'knn':knn_preds[i], 'xtrees2':xtrees2[i], 'xtrees3':xtrees3[i], 'xtrees4':xtrees4[i], 'time':time_preds[i]}
		ranked = vote(scores)
		out.append(ranked)
	return out

#""" Return an array of hashes:
#[ {'a':0.5, 'b':0.2}, {'a':1.0} ] """
#def vote(predictions):
#	weights = {'tf':0.50, 'xtrees2':0.50, 'xtrees':0.63, 'xtrees3':0.61, 'forest':0.62, 'knn':0.55, 'xtrees4':0.51, 'time':0.08}
#	combined = defaultdict(int)
#	for name,p in predictions.items():
#		weight = weights[name]
#		for index,pred in enumerate(p):
#			score = weight/(index+1)
#			combined[pred] += score
#
#	# Sort and return the top five.
#	ranked = sorted(combined.iteritems(), key=operator.itemgetter(1))
#	ranked.reverse()
#	best = []
#	for r in ranked:
#		if len(best) < 5:
#			best.append(r[0])
#	#print predictions
#	#print combined
#	#print best
#	#print "\n"
#	return best

def vote(preds):
	output = []
	model1 = preds[0]
	for query in range(len(model1)):
		all_preds = []
		for model in preds:
			model_preds = model[query]
			for p in model_preds:
				all_preds.append(p[0])
		all_preds = set(all_preds)
		scores = defaultdict(float)
		#print "---"
		#print len(preds)
		#for i,model in enumerate(preds):
		#	print "Model pred: " + str(i) + " " + str(model[query])
		for p in all_preds:
			for model in preds:
				for model_pred in model[query]:
					if model_pred[0] == p:
						scores[p] = scores[p] + float(model_pred[1])
		ranked = sorted(scores.iteritems(), key=operator.itemgetter(1))
		ranked.reverse()
		best = []
		for r in ranked:
			if len(best) < 5:
				best.append(r[0])
		#print best
		output.append(best)
	return output
