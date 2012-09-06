import re
import time
import operator
import csv

def file_to_array(file_name, ignore_line_1=True):
	output = []
	f = file(file_name)
	f = f.readlines()
	if ignore_line_1 == True:
		f.pop(0)
	for line in f:
		l = line.split(',')
		output.append(l)
	return output

# Take a file path as input.
def file_to_hash(file_name, key, value):
	output = {}
	f = file_to_array(file_name)
	for line in f:
		k = str(line[key])
		v = line[value]
		if k in output:
			old_v = output[k]
		else:
			old_v = ''
		output[k] = old_v + ' ' + v
	return output

# Accepts a hash an input.
# THe has should be keys to strings.
# It returns the hash with the strings formatted.
def format_words(data):
	output = {}
	for k,v in data.items():
		output[k] = format_string(v)
	return output

def format_string(string):
	v = re.sub("[\"']", '', string) # remove quotes
	v = v.lower() # lowercase
	v = re.sub("\s{2,}", "\s", v) # change double space to single space
	v = re.sub("(\s\n|\A\s)", '', v) # remove end/beginning of line blank space
	return v

# Accepts a hash as input.  It takes a hash of keys to strings and counts the words in each string.
# It returns the hash of key to the count
def word_count_hash(data, ngram=1):
	output = data
	for sku,v in data.items():
		output[sku] = {}
		for word in tokenize(v, ngram):
			if word in output[sku]:
				count = output[sku][word] + 1
			else:
				count = 1
			output[sku][word] = count
	return output

def slice(data, index):
	output = []
	for d in data:
		output.append(d[index])
	return output

def string_to_hash(string):
	word_count = {}
	formatted = format_string(string)
	for token in tokenize(formatted):
		if token in word_count:
			word_count[token] += 1
		else:
			word_count[token] = 1
	return word_count



###########################
######### TF-IDF ##########
###########################

def train(data, ngram=1):
	data = format_words(data)
	output = word_count_hash(data, ngram)
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
			
def train_model(csv_file, class_labels_index, input_data_index, ngram=1):
	data = file_to_hash(csv_file, class_labels_index, input_data_index)
	model = train(data, ngram)
	class_labels = slice(file_to_array(csv_file), class_labels_index)
	data = file_to_array(csv_file)
	popularity = popularity_hash(class_labels, data)
	return model, popularity

def test_data(csv_file, class_labels_index, input_data_index, items_count='All', ngram=1):
	array = file_to_array(csv_file)
	class_labels = slice(array, class_labels_index)
	test_data = slice(array, input_data_index)
	formatted_test_data = []
	for d in test_data:
		formatted = format_string(d)
		tokens = tokenize(formatted, ngram)
		formatted_test_data.append(tokens)
	if items_count != 'All':
		count = len(class_labels) - items_count
		class_labels = class_labels[count:]
		formatted_test_data = formatted_test_data[count:]
	return class_labels, formatted_test_data

def write_predictions(predictions, csv_file):
	with open(csv_file, "w") as outfile:
		writer = csv.writer(outfile, delimiter=",")
		writer.writerow(["sku"])
		for p in predictions:
			writer.writerow([" ".join(p)])

def tokenize(sentence, ngram=1):
	output = []
	tokens = sentence.split(' ')
	n_tokens = len(tokens)
	for i in xrange(n_tokens):
		for j in xrange(i+ngram, min(n_tokens, i+ngram)+1):
			joined = ' '.join(tokens[i:j])
			output.append(joined)
	return output

def real_test():
	ngram = 1
	sample_size = 'All'
	model, popularity = train_model("../data/train.csv", 1, 3, ngram)
	class_labels, td = test_data("../data/test.csv", 1, 2, sample_size, ngram)
	predictions = predict_all(td, model, popularity)
	return predictions

	

###########################
######### TF-IDF ##########
###########################

#start = time.time()
#precision, predictions = test(model, test_data, class_labels, popularity)
#print "Precision: " + str(precision) + ".\n" + str(len(test_data)) + " examples.\n Time: " + str(time.time() - start)
