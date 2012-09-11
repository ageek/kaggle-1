from numpy import *
from random import shuffle

zero = 8
one = 1
length = zero + one + one

def random_vector(vector_length):
	one_count = vector_length * 0.02
	zero_count = vector_length - one_count
	z = zeros(zero_count)
	positives = ones(one_count)
	merged = array(list(z) + list(positives))
	shuffle(merged)
	return merged

def empty_vectors(words):
	words = set(words) # unique them
	dict = {}
	for w in words:
		dict[w] = zeros(length)
	return dict

def word_vectors(corpus, negative):
	count = 0
	dict = empty_vectors(corpus)
	for word in corpus:
		if count % 10 == 0:
			context_vector = random_vector(zero, one, negative)
		dict[word] += context_vector
		count += 1
	return dict

def cosine(v1,v2):
	return float(dot(v1,v2) / (linalg.norm(v1) * linalg.norm(v2)))
