import vector
import tf_idf
from numpy import *
from datetime import datetime
from sklearn import neighbors as kNN
from sklearn import datasets
from random import shuffle
import time
import csv

# global vars
vector_length = 200
sample_size = 'All'
training = "../data/train.csv"
test = "../data/test.csv"


def random_sku_vectors(sku_array, vector_length):
	sku_vects = {}
	for sku in sku_array:
		if sku not in sku_vects:
			sku_vects[sku] = vector.random_vector(vector_length)
	return sku_vects

def sku_vectors(sku_words, word_vector_hash, vector_length):
	for pair in sku_words:
		sku = pair[0]
		word_hash = pair[1]
		vector = query_vector(word_hash, word_vector_hash, vector_length)
		pair.append(vector)
	return sku_words

def query_vector(word_hash, word_vector_hash, vector_length):
	query_vector = zeros(vector_length)
	for word,word_count in word_hash.items():
		if word in word_vector_hash:
			word_vector = word_vector_hash[word]
		else:
			word_vector = vector.random_vector(vector_length)
		for i in range(word_count):
			query_vector += word_vector
	query_vector = query_vector/(linalg.norm(query_vector))
	return query_vector

def word_vectors(csv_file, vector_length, word_to_skus=None, generated_sku_vectors=None, algorithm="normal"):
	word_vects = {}
	words_index = 3
	queries = tf_idf.slice(tf_idf.file_to_array(csv_file), words_index)
	for q in queries:
		formatted = tf_idf.format_string(q)
		for word in tf_idf.tokenize(formatted):

			if algorithm == "normal":
				if word not in word_vects:
					word_vects[word] = vector.random_vector(vector_length)

	# start of reflected
			elif algorithm == "reflected":
				word_vects[word] = zeros(vector_length)
	
	if algorithm == "reflected":
		for word in word_vects.keys():
			skus = word_to_skus[word]
			for sku in skus:
				word_vects[word] += generated_sku_vectors[sku]
	# end of reflected

	return word_vects

def word_to_sku_hash(csv_file):
	output = {}
	array = tf_idf.file_to_array(csv_file)
	text = tf_idf.slice(array, 3)
	skus = tf_idf.slice(array, 1)
	for i in range(len(text)):
		words = text[i]
		sku = skus[i]
		formatted = tf_idf.format_string(words)
		for word in tf_idf.tokenize(formatted):
			if word in output:
				output[word].append(sku)
			else:
				output[word] = [sku]
	return output
	


# return a list of vectors for each query, and a list of skus for each query,
# and a hash of words to their vectors that generated the query vectors
def data(csv_file, vector_length):
	word_vector_hash = word_vectors(csv_file, vector_length)

	# generate the sku-words
	sku_words = []
	array = tf_idf.file_to_array(csv_file)
	class_labels = tf_idf.slice(array, 1)
	text = tf_idf.slice(array, 3)
	indexes = len(text)
	for i in range(indexes):
		word_count = tf_idf.string_to_hash(text[i])
		label = class_labels[i]
		line_array = [label, word_count]
		sku_words.append(line_array)

	# get a list of only the vectors
	sku_vects = sku_vectors(sku_words, word_vector_hash, vector_length)
	vects = []
	for triplet in sku_vects:
		vect = triplet[2]
		vects.append(vect)

	return vects, class_labels, word_vector_hash

def reflected_data(vector_length):

	# generate the sku-words
	# dupliactedf rom above
	sku_words = []
	array = tf_idf.file_to_array(training)
	class_labels = tf_idf.slice(array, 1)
	text = tf_idf.slice(array, 3)
	indexes = len(text)
	for i in range(indexes):
		word_count = tf_idf.string_to_hash(text[i])
		label = class_labels[i]
		line_array = [label, word_count]
		sku_words.append(line_array)

	# new stuff
	w = word_to_sku_hash(training)
	sku_vects = random_sku_vectors(class_labels, vector_length)
	word_vects = word_vectors(training, vector_length, w, sku_vects, algorithm="reflected")
	final_sku_vects = sku_vectors(sku_words, word_vects, vector_length)


	# get a list of only the vectors
	final_sku_vects = sku_vectors(sku_words, word_vects, vector_length)
	vects = []
	for triplet in final_sku_vects:
		vect = triplet[2]
		vects.append(vect)

	return vects, class_labels, word_vects


def log(csv_file, message):
	with open(csv_file, "w") as outfile:
		writer = csv.writer(outfile, delimiter=",")
		for m in message:
			writer.writerow([" ".join(m)])
	return None

def train(csv_file, neighbors, vector_length, algorithm="normal"):
	if algorithm == "normal":
		sku_vectors, class_labels, word_vectors = data(csv_file, vector_length)
	elif algorithm == "reflected":
		sku_vectors, class_labels, word_vectors = reflected_data(vector_length)
	#else:
		#raise "You misspelled the algorithm"

	model = kNN.KNeighborsClassifier(n_neighbors=neighbors, weights='uniform')
	model.fit(sku_vectors, class_labels)
	return model, word_vectors, sku_vectors, class_labels

def test(model, neighbors, test_data, class_labels,  sample_size, vector_length):
	correct = 0.
	start = time.time()
	for i in range(sample_size):
		prediction = model.predict(test_data[i])[0]
		if int(prediction) == int(class_labels[i]):
			correct += 1.
	return correct/sample_size

# For testing the test function with varying parameters
def evaluate():
	out = []
	for i in range(10):
		vector_length = 50 + (i * 50)
		precision = test(20, 42365, vector_length)
		out.append([str(vector_length), str(precision)])
	log("out2.csv", out)

def real_test():
	vector_length = 100
	neighbors = 20
	output = []
	model, word_vectors = train("../data/train.csv", neighbors, vector_length)
	queries = tf_idf.slice(tf_idf.file_to_array("../data/test.csv"), 2)
	query_vctors = []
	for q in queries:
		word_hash = tf_idf.string_to_hash(tf_idf.format_string(q))
		vect = query_vector(word_hash, word_vectors, vector_length)
		pred = model.predict(vect)
		output.append([pred])
	return output

n = 20
sample_size = 42365
model, word_vects, sku_vectors, class_labels = train(training, n, vector_length, algorithm="reflected")
print test(model, n, sku_vectors, class_labels, sample_size, vector_length)
