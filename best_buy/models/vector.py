from numpy import *
from random import shuffle

zero = 8
one = 1
length = zero + one + one

def random_vector(zero, one, negative=False):
	z = zeros(zero)
	positives = ones(one)
	if negative == True:
		negatives = ones(one) - 2
		merged = array(list(z) + list(positives) + list(negatives))
	elif negative == False:
		merged = array(list(z) + list(positives) + list(positives))
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
	return float(dot(v1,v2) / (norm(v1) * norm(v2)))
