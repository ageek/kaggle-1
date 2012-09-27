import operator

# this returns the input for the most_popular() function.
def popularity_hash(skus, file_array):
	output = {}
	for line in file_array:
		sku = line[1]
		if sku in output:
			output[sku] += 1
		else:
			output[sku] = 1
	return output

def most_popular(skus, popularity):
	winner, best, best_score = 'failure', 0, 0
	#for sku,score in skus.items():
	for sku in skus:
		clicks = popularity[sku]
		if clicks >= best:
			winner = sku
			#best_score = score
	return winner #{winner: best_score}
			
def sort_by_popularity(skus, popularity):
	skus = set(skus)
	rank = {}
	for s in skus:
		rank[s] = popularity[s]
	sorted_scores = sorted(rank.iteritems(), key=operator.itemgetter(1))
	sorted_scores.reverse()

	output = []
	for s in sorted_scores:
		output.append(s[0])
	return output
