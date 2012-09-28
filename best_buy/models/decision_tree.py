from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import neighbors as kNN
import time
import knn
import sys
sys.path.append("../../")
import kaggle

vector_length = 100
extra = "../data/extra.csv"
training = "../data/train.csv"
testing = "../data/test.csv"
name = "../data/sku_names.csv"

def train_tree():
	sku_vectors, class_labels, word_vectors, sku_hash = knn.data(training, vector_length, "All")

	xtrees = ExtraTreesClassifier(n_estimators=7, max_depth=None, min_samples_split=1, random_state=0)
	model1 = xtrees.fit(sku_vectors, class_labels)

	forest = RandomForestClassifier(n_estimators=5, max_depth=None, min_samples_split=1, random_state=0)
	model2 = forest.fit(sku_vectors, class_labels)

	neigh = kNN.KNeighborsClassifier(n_neighbors=10)
	model3 = neigh.fit(sku_vectors, class_labels)

	models = [model1, model2, model3]
	return models, word_vectors

def validation_test():
	models, word_vectors = train_tree()
	# Create an array for each model to store predictions.
	output = []
	for m in models:
		output.append([])

	file_array = kaggle.file_to_array(testing, "All")
	queries = kaggle.slice(file_array, 2)
	skus = kaggle.slice(file_array, 1)
	total = 0.
	correct = 0.
	for index,q in enumerate(queries):
		word_hash = kaggle.string_to_hash(kaggle.format_string(q))
		vect = knn.query_vector(word_hash, word_vectors, vector_length)
		preds = []
		for i,m in enumerate(models):
			pred = m.predict(vect)[0]
			pred = str(int(pred))
			preds.append(pred)
			output[i].append([pred])

		# For testing accuracy.
		#sub_out = list(set(preds))
		#if len(sub_out) > 1:
			#sub_out = []
		#total += len(sub_out)

		#correct_sku = skus[index]
		#if correct_sku in sub_out:
			#correct += 1.
	
	#answered = total/len(queries)
	#print "Correct: " + str(correct)
	#precision = correct/total
	recall = correct/len(queries)
	#print "Answered: " + str(answered)
	#print "Precision: " + str(precision)
	#print "Recall: " + str(recall)
	return recall, output

def real_test():
	score, preds = validation_test()
	return preds[0], preds[1], preds[2]

#s = time.time()
#score, preds = validation_test()
#print "Recall: " + str(score)
#print "Time: " + str(time.time() - s)
