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
#vector_length = 50
sample_size = 'All'



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

def word_vectors(csv_file, vector_length):
	word_vects = {}
	words_index = 3
	queries = tf_idf.slice(tf_idf.file_to_array(csv_file), words_index)
	for q in queries:
		formatted = tf_idf.format_string(q)
		for word in tf_idf.tokenize(formatted):
			if word not in word_vects:
				word_vects[word] = vector.random_vector(vector_length)
	return word_vects


# return a list of vectors for each query, and a list of skus for each query,
# and a hash of words to their vectors that generated the query vectors
def data(csv_file, vector_length):
	ngram = 1
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

def log(csv_file, message):
	with open(csv_file, "w") as outfile:
		writer = csv.writer(outfile, delimiter=",")
		for m in message:
			writer.writerow([" ".join(m)])
	return None

def train(csv_file, neighbors, vector_length):
	sku_vectors, class_labels, word_vectors = data(csv_file, vector_length)
	clf = kNN.KNeighborsClassifier(n_neighbors=neighbors, weights='uniform')
	clf.fit(sku_vectors, class_labels)
	return clf, word_vectors

def test(model, neighbors, sample_size, vector_length):
	clf = train("../data/train.csv")
	correct = 0.
	start = time.time()
	for i in range(sample_size):
		prediction = model.predict(sku_vectors[i])[0]
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
	vector_length = 200
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

print real_test()
