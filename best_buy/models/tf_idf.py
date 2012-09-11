import re
import time
import operator
import sys
sys.path.append("../../")
import kaggle
import popular


def train(data, ngram=1):
	data = kaggle.format_words(data)
	output = kaggle.word_count_hash(data, ngram)
	return output

def classify(words, model, popularity):
	top_items = 5 # Number of predictions to return
	scores = score_all(words, model, popularity)
	winner, best = [], 0.0
	sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1))
	sorted_scores.reverse()
	sorted_scores_list = []
	for x in sorted_scores[0:top_items]:
		sorted_scores_list.append(x[0])
	return sorted_scores_list

#def test(model, 
def score_all(query_words, model, popularity):
	output = {}
	s = 0.
	idf = inverse_document_frequency(model, query_words, debug=True)
	for sku,target_words in model.items():
		s = score(query_words, target_words, idf)
		output[sku] = s
	return output

def score(query, target, idf, debug=False):
	output = 0
	for q in query:
		tf = term_frequency(q, target)
		if q in idf:
			word_idf = idf[q]
		else:
			word_idf = 1.
		if word_idf == 0.:
			word_idf = 1.

		tf_idf = tf/word_idf
		output += tf_idf
	return output

def term_frequency(query, target):
	total_words = 0
	for word,count in target.items():
		total_words += count
	# in case a higher n-gram doesn't have anything that long
	if total_words == 0:
		total_words = 1
	if query in target:
		word_count = target[query]
	else:
		word_count = 0
	return float(word_count)/total_words

def inverse_document_frequency(model, words, debug=False):
	idf = {}
	for w in words:
		idf[w] = 0.
	
	for sku,document in model.items():
		for w in words:
			if w in document:
				idf[w] += 1

	new_idf = {}
	document_count = len(model)
	for word,count in idf.items():
		new_idf[word] = count/document_count
	return new_idf

# get the predictions
def predict_all(data, model, popularity):
	output = []
	for d in data:
		prediction = classify(d, model, popularity)
		output.append(prediction)
	return output


# Returns the percent that were correct and the predictions.
def test(model, data, class_labels, popularity):
	predictions = predict_all(data, model)
	
	# see how many are correct
	correct = 0.
	total_preds = 0.
	for index,prediction in enumerate(predictions):
		correct_answer = class_labels[index]
		total_preds += len(prediction)
		if correct_answer in prediction:
			correct += 1.
	print "Average predictions: " + str(total_preds/len(predictions))
	return correct/len(data), predictions

def train_model(csv_file, class_labels_index, input_data_index, ngram=1):
	data = kaggle.file_to_hash(csv_file, class_labels_index, input_data_index)
	model = train(data, ngram)
	class_labels = kaggle.slice(kaggle.file_to_array(csv_file), class_labels_index)
	data = kaggle.file_to_array(csv_file)
	popularity = popular.popularity_hash(class_labels, data)
	return model, popularity

def test_data(csv_file, class_labels_index, input_data_index, items_count='All', ngram=1):
	array = kaggle.file_to_array(csv_file)
	class_labels = kaggle.slice(array, class_labels_index)
	test_data = kaggle.slice(array, input_data_index)
	formatted_test_data = []
	for d in test_data:
		formatted = kaggle.format_string(d)
		tokens = kaggle.tokenize(formatted, ngram)
		formatted_test_data.append(tokens)
	if items_count != 'All':
		count = len(class_labels) - items_count
		class_labels = class_labels[count:]
		formatted_test_data = formatted_test_data[count:]
	return class_labels, formatted_test_data

def real_test():
	ngram = 1
	sample_size = 'All'
	model, popularity = train_model("../data/train.csv", 1, 3, ngram)
	class_labels, td = test_data("../data/test.csv", 1, 2, sample_size, ngram)
	predictions = predict_all(td, model, popularity)
	return predictions

	

#start = time.time()
#real_test()
#print time.time() - start
#precision, predictions = test(model, test_data, class_labels, popularity)
#print "Precision: " + str(precision) + ".\n" + str(len(test_data)) + " examples.\n Time: " + str(time.time() - start)
