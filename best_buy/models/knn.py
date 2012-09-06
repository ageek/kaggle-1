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
vector_length = 50
sample_size = 'All'



def sku_vectors(sku_words, word_vector_hash):
	for pair in sku_words:
		sku = pair[0]
		word_hash = pair[1]
		vector = query_vector(word_hash, word_vector_hash)
		pair.append(vector)
	return sku_words

def query_vector(word_hash, word_vector_hash):
	query_vector = zeros(vector_length)
	for word,word_count in word_hash.items():
		word_vector = word_vector_hash[word]
		for i in range(word_count):
			query_vector += word_vector
	query_vector = query_vector/(linalg.norm(query_vector))
	return query_vector

def word_vectors(csv_file):
	word_vects = {}
	words_index = 3
	queries = tf_idf.slice(tf_idf.file_to_array(csv_file), words_index)
	for q in queries:
		formatted = tf_idf.format_string(q)
		for word in tf_idf.tokenize(formatted):
			if word not in word_vects:
				word_vects[word] = vector.random_vector(40, 5)
	return word_vects


# return a list of vectors for each query, and a list of skus for each query,
# and a hash of words to their vectors that generated the query vectors
def data(csv_file):
	ngram = 1
	word_vector_hash = word_vectors(csv_file)

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
	sku_vects = sku_vectors(sku_words, word_vector_hash)
	vects = []
	for triplet in sku_vects:
		vect = triplet[2]
		vects.append(vect)

	return vects, class_labels, word_vector_hash

def test(csv_file):
	data(csv_file)

def log(csv_file, message):
	with open(csv_file, "w") as outfile:
		writer = csv.writer(outfile, delimiter=",")
		for m in message:
			writer.writerow([" ".join(m)])
	return None

def test(neighbors, sample_size):
	sku_vectors, class_labels, word_vectors = data("../data/train.csv")
	clf = kNN.KNeighborsClassifier(n_neighbors=5, weights='uniform')
	clf.fit(sku_vectors, class_labels)

	correct = 0.
	start = time.time()
	for i in range(sample_size):
		prediction = clf.predict(sku_vectors[i])[0]
		if int(prediction) == int(class_labels[i]):
			correct += 1.
	return correct/sample_size

def main(csv_file="out.csv"):
	n = 1
	out = []
	sku_vectors, class_labels, word_vectors = data("../data/train.csv")
	for i in range(5):
		clf = kNN.KNeighborsClassifier(n_neighbors=i, weights='uniform')
		clf.fit(sku_vectors, class_labels)
		correct = 0.
		sample_size = 5#42365
		start = time.time()
		for i in range(sample_size):
			prediction = clf.predict(sku_vectors[i])[0]
			if int(prediction) == int(class_labels[i]):
				correct += 1.
		precision = correct/sample_size
		print precision
		out.append([n, precision])
		i += 2

	log(csv_file,out)
	return None




out = []
for i in range(5):
	n = 3 + (i * 3)
	precision = test(n, 42365)
	out.append([str(n), str(precision)])
log("out.csv", out)



#test_vect = query_vector({'gears':1, 'of':1, 'war':1}, word_vectors)
#test_vect2 = query_vector({'xbox': 1, '360':1}, word_vectors)
#test_vect3 = query_vector({'alice':1, 'madness':1, 'returns':1}, word_vectors)
#test_vect4 = query_vector({'you':1, 'know':1, 'jack':1}, word_vectors)
#print correct/sample_size
#print "Time: " + str(time.time() - start)

