from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from numpy import *
import operator
import time
import numpy as np
from numpy import *

				
class ConfidenceDecorator:

	def __init__(self, decorated, data, labels):
		self.decorated = decorated
		self.confidence_model = self.__train_confidence__(data, labels)
		print "Confidence: " + str(self.__test_confidence__(data, labels))

	def __train_confidence__(self, data, labels, debug=False):
		training, training_labels = [], []
		for index,d in enumerate(data[0:1000]):
			answer = labels[index]
			preds = self.predictions(d)
			first_pred = preds[0][0]
			scores = []
			for p in preds:
				scores.append(round(p[1], 3))
			training.append(scores)
			if debug == True:
				print first_pred == self.decorated.predict(d)[0]
			if first_pred == answer:
				training_labels.append(1)
			else:
				training_labels.append(0)
		regression = LogisticRegression(C=1000., penalty='l2', tol=0.01)
		training, training_labels = array(training), array(training_labels)
		if debug == True:
			print "Data for regression:\n" + str(training)
			print "Labels for regression: " + str(training_labels)
		regression.fit(training, training_labels)
		return regression
	
	def __test_confidence__(self, training, training_labels):
		correct = 0.
		half = len(training)/2
		training = training[half:]
		training_labels = training_labels[half:]
		for i,t in enumerate(training):
			predictions = self.predictions(t)
			pred, confidence = predictions[0][0], predictions[0][1]
			if pred == training_labels[i] and confidence >= 0.5:
				correct += 1.
			elif pred != training_labels[i] and confidence < 0.5:
				correct += 1.
		return correct/len(training)

	def predict(self, arg):
		return self.decorated.predict(arg)
	
	def predictions(self, arg):
		probs = self.decorated.predict_proba(arg)[0]
		scores = {}
		for index,prob in enumerate(probs):
			label = self.decorated.classes_[index]
			scores[label] = prob
		ranked = sorted(scores.iteritems(), key=operator.itemgetter(1))
		ranked.reverse()
		return ranked
	
	def confidence(self, arg):
		preds = self.predictions(arg)
		scores = []
		for pred in preds:
			label = pred[0]
			score = pred[1]
			scores.append(score)
		output = self.confidence_model.predict_proba(scores)
		return output[0][1]


#data = array([[0,1,1],[1,1,0],[1,1,1],[1,1,1], [1,0,0],[1,0,0],[1,1,1],[1,1,1], [1,0,1],[1,0,1],[1,1,1],[1,1,1], [0,1,1], [1,0,1], [1,0,1],[1,1,1]])
#classes = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

#xtrees = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
#model = xtrees.fit(data, classes)
#print dir(model)
#print model.classes_
#d = ConfidenceDecorator(model, data, classes)
#x = [1,0,1]
#x = [0,1,1]
#x = [1,0,0]
#print "Input: " + str(x)
#print "Predcion: " + str(d.predict(x))
#print "Confidence: " + str(d.confidence(x))
#print dir(d.confidence_model)

#s = time.time()
#for i in range(1000):
	#model.predict_proba(x)
	#d.predictions(x)
#	d.confidence(x)
#print time.time() - s
