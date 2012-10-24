import sys
x = sys.path[1]
sys.path.append("/home/I829287/lib/python/")
sys.path.remove(x)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import neighbors
import time
sys.path.append("../../")
import kaggle
import knn
import best_buy
import ada_boost
from decorator import ConfidenceDecorator

extra = "../data/extra.csv"
training = "../data/train.csv"
adapt1 = "../data/adapt_2.csv"
adapt2 = "../data/adapt_3.csv"
adapt3 = "../data/adapt_4.csv"
testing = "../data/test.csv"
name = "../data/sku_names.csv"
vector_length = 1000

def train_tree():
	word_vector_hash = knn.word_vectors(training, vector_length, False)

	sku_vectors, class_labels, _, sku_hash = knn.data(adapt1, vector_length, 'all', word_vector_hash)
	xtrees = ExtraTreesClassifier(n_estimators=1, max_depth=None, min_samples_split=1, random_state=0)
	model2 = xtrees.fit(sku_vectors, class_labels)

	sku_vectors, class_labels, _, sku_hash = knn.data(adapt2, vector_length, 'all', word_vector_hash)
	xtrees = ExtraTreesClassifier(n_estimators=1, max_depth=None, min_samples_split=1, random_state=0)
	model3 = xtrees.fit(sku_vectors, class_labels)

	sku_vectors, class_labels, _, sku_hash = knn.data(adapt3, vector_length, 'all', word_vector_hash)
	xtrees = ExtraTreesClassifier(n_estimators=1, max_depth=None, min_samples_split=1, random_state=0)
	model4 = xtrees.fit(sku_vectors, class_labels)

	# Non-adaptive data
	sku_vectors, class_labels, _, sku_hash = knn.data(training, vector_length, False, word_vector_hash)
	model2 = ConfidenceDecorator(model2, sku_vectors, class_labels)
	model3 = ConfidenceDecorator(model3, sku_vectors, class_labels)
	model4 = ConfidenceDecorator(model4, sku_vectors, class_labels)

	xtrees = ExtraTreesClassifier(n_estimators=1, max_depth=None, min_samples_split=1, random_state=0)
	model1 = xtrees.fit(sku_vectors, class_labels)
	model1 = ConfidenceDecorator(model1, sku_vectors, class_labels)

	forest = RandomForestClassifier(n_estimators=3, max_depth=None, min_samples_split=1, random_state=0)
	model5 = forest.fit(sku_vectors, class_labels)
	model5 = ConfidenceDecorator(model5, sku_vectors, class_labels)

	#neigh = neighbors.KNeighborsClassifier(n_neighbors=10, warn_on_equidistant=False, weights="distance")
	#model6 = neigh.fit(sku_vectors, class_labels)
	#model6 = ConfidenceDecorator(model6, sku_vectors, class_labels)

	models = [model1, model2, model3, model4, model5]# model6]
	return models, word_vector_hash

def validation_test():
	models, word_vectors = train_tree()
	correct_data = best_buy.sku_to_searches()
	# Create an array for each model to store predictions.
	output = []
	for m in models:
		output.append([])

	file_array = kaggle.file_to_array(training, True)
	queries = kaggle.slice(file_array, 3)
	skus = kaggle.slice(file_array, 1)
	total = 0.
	correct = 0.
	correct_pop = 0.
	wrong_pop = 0.

	right_answers = []
	wrong_answers = []

	for index,q in enumerate(queries):
		word_hash = kaggle.string_to_hash(kaggle.format_string(q))
		vect = knn.query_vector(word_hash, word_vectors, vector_length)
		all_preds = []
		for i,m in enumerate(models):
			preds = m.predictions(vect)[0:5]
			output[i].append(preds)
			pred = []
			for p in preds:
				pred.append(p[0])
				all_preds.append(p[0])

		#For testing accuracy.
		sub_out = set(all_preds)
		total += len(sub_out)

		correct_sku = skus[index]
		pop = len(correct_data[correct_sku])
		if correct_sku in sub_out:
			correct += 1.
			correct_pop += pop
			right_answers.append([q,correct_sku])
		else:
			wrong_pop += pop
			wrong_answers.append([q,correct_sku])
			#print "\nQuery: " + q
			#print "Correct Answer: " + str(correct_data[correct_sku][0:6]) + ". Popularity: " + str(len(correct_data[correct_sku]))
			#for p in sub_out:
			#	print "Prediction: " + str(correct_data[p][0:6]) + ". Popularity: " + str(len(correct_data[p]))
	
	wrong = len(queries) - correct
	#print "Avg wrong pop: " + str(wrong_pop/wrong)
	#print "Avg correct pop: " + str(correct_pop/correct)
	answered = total/len(queries)
	#print "Correct: " + str(correct)
	precision = correct/total
	recall = correct/len(queries)
	print "Total: " + str(total)
	#print "Answered: " + str(answered)
	print "Precision: " + str(precision)
	print "Recall: " + str(recall)
	return output

def real_test():
	preds = validation_test()
	return preds

#for i in range(1):
#	print "------"
#	print vector_length
#	s = time.time()
#	preds = validation_test()
#	print "Time: " + str(time.time() - s)

#ada_boost.adapt(right, wrong)
