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



######### TF-IDF ##########

def train(data):
	data = format_words(data)
	output = word_count_hash(data)
	return output

def classify(words, model):
	scores = score_all(words, model)
	winner, best = 'unknown', 0.0
	for sku,score in scores.items():
		if score >= best:
			best = score
			winner = sku
	return winner, best

#def test(model, 
def score_all(query_words, model):
	output = {}
	for sku,target_words in model.items():
		output[sku] = score(query_words, target_words)
	return output

def score(query, target):
	output = 0
	for q in query:
		output += term_frequency(q, target)
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

def test(model, data, class_labels):
	# get the predictions
	predictions = []
	for d in data:
		prediction, score = classify(d, model)
		print "predicted: " + str(d) + " EQUALS " + str(model[prediction]) + ". With a score of: " + str(score)
		predictions.append(prediction)

	# see how many are correct
	correct = 0.
	for index,prediction in enumerate(predictions):
		if prediction == class_labels[index]:
			print 'correct'
			correct += 1.
	return correct/len(data)
	

######### TF-IDF ##########

data = file_to_hash("../data/train.csv", 0, 3)
model = train(data)

array = file_to_array("../data/train.csv")
class_labels = slice(array, 0)
test_data = slice(array, 3)
formatted = []
for d in test_data:
	formatted.append(format_string(d))

start = time.time()
results = test(model, formatted[42264:], class_labels[42264:])
print "Precision: " + str(results)
print "Time: " + str(time.time() - start)
