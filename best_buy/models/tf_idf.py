import re
import time
import operator
import sys
sys.path.append("../../")
import kaggle
import popular

training = "../data/train.csv"
extra = "../data/sku_names.csv"
testing = "../data/test.csv"

def train(data, ngram=1):
	data = kaggle.format_words(data)
	output = kaggle.word_count_hash(data, ngram)
	return output

def classify(words, model, popularity):
	top_items = 10 # Number of predictions to return
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

def voted_predict_all(data, model1, model2, popularity, w):
	predictions1 = predict_all(data, model1, popularity)
	predictions2 = predict_all(data, model2, popularity)
	predictions = vote(predictions1, predictions2, popularity, w)
	return predictions


# Returns the percent that were correct and the predictions.
def test(model1, model2, data, class_labels, popularity):
	predictions = voted_predict_all(data, model1, model2, popularity)
	
	# see how many are correct
	sum_score = 0.
	for index,prediction in enumerate(predictions):
		correct_answer = class_labels[index]
		if correct_answer in prediction:
			score = 1.0/(prediction.index(correct_answer)+1)
			sum_score += score
	predictions_count = float(len(predictions))
	score = sum_score/predictions_count
	return score, predictions

def train_model(csv_file, ngram=1, validation=False):
	class_labels_index = 1
	input_data_index = 3
	data = kaggle.file_to_hash(csv_file, class_labels_index, input_data_index, validation)
	model = train(data, ngram)
	data = kaggle.file_to_array(csv_file, validation)
	class_labels = kaggle.slice(data, class_labels_index)
	popularity = popular.popularity_hash(class_labels, data)
	return model, popularity

def test_data(csv_file, class_labels_index, input_data_index, validation, items_count, ngram=1):
	array = kaggle.file_to_array(csv_file, validation)
	class_labels = kaggle.slice(array, class_labels_index)
	test_data = kaggle.slice(array, input_data_index)
	formatted_test_data = []
	for d in test_data:
		formatted = kaggle.format_string(d)
		tokens = kaggle.tokenize(formatted, ngram)
		formatted_test_data.append(tokens)
	if items_count != 'All':
		class_labels, formatted_test_data = class_labels[0:items_count], formatted_test_data[0:items_count]
	return class_labels, formatted_test_data

def vote(pred1, pred2, popularity, weight):
	output = []

	uni_gram_weight = 1.0
	bi_gram_weight = 0.5#0.3
	popular_weight = 0.5
	for index,preds in enumerate(pred1):
		scored = {}
		pop = popular.sort_by_popularity(preds + pred2[index], popularity)

		length = len(preds)
		#print preds
		for i,p in enumerate(preds):
			score = (1 - (float(i)/length)) * uni_gram_weight
			if p in scored:
				scored[p] += score
			elif p not in scored:
				scored[p] = score
		#print pred2[index]
		for i,p in enumerate(pred2[index]):
			score = (1 - (float(i)/length)) * bi_gram_weight
			if p in scored:
				scored[p] += score
			elif p not in scored:
				scored[p] = score
		for i,p in enumerate(pop):
			score = (1 - (float(i)/length)) * popular_weight
			if p in scored:
				scored[p] += score
			elif p not in scored:
				scored[p] = score
		sorted_scores = sorted(scored.iteritems(), key=operator.itemgetter(1))
		sorted_scores.reverse()
		#print sorted_scores
		sub_out = []
		for x in sorted_scores[0:5]:
			sub_out.append(x[0])
		output.append(sub_out)
		#print sub_out
	return output

		

def real_test(w):
	sample_size = 'All'
	model1, popularity = train_model(training, ngram=1, validation='all')
	#_, popularity = train_model(training, ngram=1, validation='all')
	model2, _ = train_model(extra, ngram=1, validation='all')

	_, td = test_data(testing, 1, 2, 'all', sample_size)
	predictions = voted_predict_all(td, model1, model2, popularity, w)
	return predictions

def validation_test():
	start = time.time()
	sample_size = 'All'
	model1, popularity = train_model(training, ngram=1, validation=False)
	_, popularity = train_model(training, ngram=1, validation="all")
	model2, _ = train_model(extra, ngram=1, validation="all")
	
	validation = True
	class_labels, _test_data = test_data(training, 1, 3, validation, sample_size)
	precision, predictions = test(model1, model2, _test_data, class_labels, popularity)
	return "Precision: " + str(precision) + ".\n" + str(len(_test_data)) + " examples.\n Time: " + str(time.time() - start)
