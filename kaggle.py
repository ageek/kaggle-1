import re
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

def tokenize(sentence, ngram=1):
	output = []
	tokens = sentence.split(' ')
	n_tokens = len(tokens)
	for i in xrange(n_tokens):
		for j in xrange(i+ngram, min(n_tokens, i+ngram)+1):
			joined = ' '.join(tokens[i:j])
			output.append(joined)
	return output

def write_predictions(predictions, csv_file):
	with open(csv_file, "w") as outfile:
		writer = csv.writer(outfile, delimiter=",")
		writer.writerow(["sku"])
		for p in predictions:
			writer.writerow([" ".join(p)])
