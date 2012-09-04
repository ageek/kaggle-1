import re
import time

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
def word_count_hash(data):
	output = data
	for sku,v in data.items():
		output[sku] = {}
		for word in v.split(' '):
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



###########################
######### TF-IDF ##########
###########################

def train(data):
	data = format_words(data)
	output = word_count_hash(data)
	return output

def classify(words, model, popularity):
	scores = score_all(words, model, popularity)
	winner, best = [], 0.0
	for sku,sc in scores.items():
		if sc > best:
			best = sc
			winner = [sku]
		elif sc == best:
			winner.append(sku)
	output = most_popular(winner, popularity)
	#output = {output.keys()[0]: 1.}
	return [output], best

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
		tf_idf = tf/idf[q]
		output += tf_idf
		if debug == True:
			print "TF: " + str(tf) + " " + q + ". Target: " + str(target)
	return output

def term_frequency(query, target):
	total_words = 0
	for word,count in target.items():
		total_words += count

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

def test(model, data, class_labels, popularity):
	# get the predictions
	predictions = []
	for d in data:
		prediction, score = classify(d, model, popularity)
		predictions.append(prediction)

	# see how many are correct
	correct = 0.
	total_preds = 0.
	for index,prediction in enumerate(predictions):
		correct_answer = class_labels[index]
		#print str(prediction) + " for " + str(model[correct_answer])
		total_preds += len(prediction)
		if correct_answer in prediction:
			correct += 1.
	print "Average predictions: " + str(total_preds/len(predictions))
	return correct/len(data)

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
			

###########################
######### TF-IDF ##########
###########################


################# training data ###################
data = file_to_hash("../data/train.csv", 1, 3)
model = train(data)

data = file_to_array("../data/train.csv")
popularity = popularity_hash(class_labels, data)
#formatted_test_data = formatted_test_data[41365:]
#class_labels = class_labels[41365:]
################# training data ###################



################# testing data ####################
array = file_to_array("../data/test.csv")
class_labels = slice(array, 1)
test_data = slice(array, 2)
formatted_test_data = []
for d in test_data:
	formatted_test_data.append(format_string(d).split(' '))
################# testing data ####################


start = time.time()
results = test(model, formatted_test_data, class_labels, popularity)
print "Precision: " + str(results)
print str(len(formatted_test_data)) + " examples."
print "Time: " + str(time.time() - start)
