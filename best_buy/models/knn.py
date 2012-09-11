import tf_idf
from numpy import *
from datetime import datetime
from sklearn import neighbors as kNN
from sklearn import datasets
from random import shuffle
import operator
import time
import csv
import sys
sys.path.append("../../")
import kaggle
import vector

# global vars
vector_length = 200
sample_size = 'All'
training = "../data/train.csv"
testing = "../data/test.csv"


def reflected_sku_vectors():
	array = kaggle.file_to_array(training)
	users = kaggle.slice(array, 0)
	skus = kaggle.slice(array, 1)
	user_vects = {}
	sku_users = {}
	sku_vects = {}
	for i in range(len(users)):
		u = users[i]
		sku = skus[i]
		if u not in user_vects:
			user_vects[u] = vector.random_vector(vector_length)

		if sku in sku_users:
			sku_users[sku].append(u)
		elif sku not in sku_users:
			sku_users[sku] = [u]
			sku_vects[sku] = zeros(vector_length)

	for sku,users in sku_users.items():
		for u in users:
			sku_vects[sku] += user_vects[u]

	return sku_vects

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

def sku_vector_hash(sku_words, word_vector_hash, vector_length):
	output = {}
	for pair in sku_words:
		sku = pair[0]
		word_hash = pair[1]
		vector = query_vector(word_hash, word_vector_hash, vector_length)
		output[sku] = vector
	return output

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
	queries = kaggle.slice(kaggle.file_to_array(csv_file), words_index)
	for q in queries:
		formatted = kaggle.format_string(q)
		for word in kaggle.tokenize(formatted):

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
	array = kaggle.file_to_array(csv_file)
	text = kaggle.slice(array, 3)
	skus = kaggle.slice(array, 1)
	for i in range(len(text)):
		words = text[i]
		sku = skus[i]
		formatted = kaggle.format_string(words)
		for word in kaggle.tokenize(formatted):
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
	array = kaggle.file_to_array(csv_file)
	class_labels = kaggle.slice(array, 1)
	text = kaggle.slice(array, 3)
	indexes = len(text)
	for i in range(indexes):
		word_count = kaggle.string_to_hash(text[i])
		label = class_labels[i]
		line_array = [label, word_count]
		sku_words.append(line_array)

	# get a list of only the vectors
	sku_vects = sku_vectors(sku_words, word_vector_hash, vector_length)
	vects = []
	for triplet in sku_vects:
		vect = triplet[2]
		vects.append(vect)

	sku_hash = sku_vector_hash(sku_words, word_vector_hash, vector_length)

	return vects, class_labels, word_vector_hash, sku_hash

def reflected_data(vector_length):

	# generate the sku-words
	# dupliactedf rom above
	sku_words = []
	array = kaggle.file_to_array(training)
	class_labels = kaggle.slice(array, 1)
	text = kaggle.slice(array, 3)
	indexes = len(text)
	for i in range(indexes):
		word_count = kaggle.string_to_hash(text[i])
		label = class_labels[i]
		line_array = [label, word_count]
		sku_words.append(line_array)

	# new stuff
	w = word_to_sku_hash(training)
	#sku_vects = random_sku_vectors(class_labels, vector_length) # without user reflection
	sku_vects = reflected_sku_vectors()# with user reflection
	word_vects = word_vectors(training, vector_length, w, sku_vects, algorithm="reflected")
	final_sku_vects = sku_vectors(sku_words, word_vects, vector_length)


	# get a list of only the vectors
	final_sku_vects = sku_vectors(sku_words, word_vects, vector_length)
	vects = []
	for triplet in final_sku_vects:
		vect = triplet[2]
		vects.append(vect)

	sku_hash = sku_vector_hash(sku_words, word_vector_hash, vector_length)

	return vects, class_labels, word_vects, sku_hash


def log(csv_file, message):
	with open(csv_file, "w") as outfile:
		writer = csv.writer(outfile, delimiter=",")
		for m in message:
			writer.writerow([" ".join(m)])
	return None

def train(csv_file, neighbors, vector_length, algorithm="normal"):
	if algorithm == "normal":
		sku_vectors, class_labels, word_vectors, sku_hash = data(csv_file, vector_length)
	elif algorithm == "reflected":
		sku_vectors, class_labels, word_vectors, sku_hash = reflected_data(vector_length)
	model = kNN.KNeighborsClassifier(n_neighbors=neighbors, weights='uniform')
	model.fit(sku_vectors, class_labels)
	return model, word_vectors, sku_vectors, class_labels, sku_hash

def test(model, neighbors, test_data, class_labels, sample_size, vector_length, sku_hash):
	correct = 0.
	start = time.time()
	for i in range(sample_size):
		prediction = model.predict(test_data[i])[0]
		p_vector = sku_hash[str(prediction)]
		predictions = nearest_skus(p_vector, sku_hash)
		#if int(prediction) == int(class_labels[i]):
		for p in predictions:
			if int(p) == int(class_labels[i]):
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
	model, word_vectors, _, _2, sku_hash = train(training, neighbors, vector_length, algorithm="normal")
	queries = kaggle.slice(kaggle.file_to_array(testing), 2)
	for q in queries:
		word_hash = kaggle.string_to_hash(kaggle.format_string(q))
		vect = query_vector(word_hash, word_vectors, vector_length)
		pred = model.predict(vect)[0]
		p_vector = sku_hash[str(pred)]
		predictions = nearest_skus(p_vector, sku_hash)
		output.append(predictions)
	print output
	return output

def nearest_skus(search, sku_vectors):
	top_items = 5
	distances = {}
	for sku,sku_vector in sku_vectors.items():
		distances[sku] = vector.cosine(search, sku_vector)
	sorted_dist = sorted(distances.iteritems(), key=operator.itemgetter(1))
	sorted_dist.reverse()
	sorted_dist_list = []
	for x in sorted_dist[0:top_items]:
		sorted_dist_list.append(str(x[0]))
	return sorted_dist_list

	

#n = 20
##sample_size = 42365
#sample_size = 100
#
#model, word_vectors, sku_vectors, labels, sku_hash = train(training, n, vector_length, algorithm="normal")
#queries = kaggle.slice(kaggle.file_to_array(training), 3)
#test_data = []
#for q in queries:
#	word_hash = kaggle.string_to_hash(kaggle.format_string(q))
#	test_data.append(query_vector(word_hash, word_vectors, vector_length))
#
#print test(model, n, test_data, labels, sample_size, vector_length, sku_hash)





#csv_file = training
#word_vector_hash = word_vectors(csv_file, vector_length)
#sku_words = []
#array = kaggle.file_to_array(csv_file)
#class_labels = kaggle.slice(array, 1)
#text = kaggle.slice(array, 3)
#indexes = len(text)
#for i in range(indexes):
#	word_count = kaggle.string_to_hash(text[i])
#	label = class_labels[i]
#	line_array = [label, word_count]
#	sku_words.append(line_array)
#
#sku_hash = sku_vector_hash(sku_words, word_vector_hash, vector_length)
#v = sku_hash['2807036']
#print vector.cosine(v,v)
