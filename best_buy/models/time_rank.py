import sys
sys.path.append("../../")
import kaggle
import time
import re
from xml.dom.minidom import parseString
from time import strptime
from time import mktime
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
from numpy import *
from decorator import ConfidenceDecorator

training = "../data/train.csv"
testing = "../data/test.csv"

def skus(csv):
	return kaggle.slice(csv, 1)

def query_times(csv, column):
	times = kaggle.slice(csv, column)
	unix = []
	for t in times:
		unix.append([unix_time(t)])
	return unix
	
def unix_time(t):
	t = re.sub("\"", '', t)
	t = re.sub("\.\d*", '', t)
	date = t.split(" ")	
	formatted = []
	formatted += date[0].split("-")
	formatted += date[1].split(":")
	string = " ".join(formatted)
	struct_time = strptime(string, "%Y %m %d %H %M %S")
	dt = datetime.fromtimestamp(mktime(struct_time))
	return int(dt.strftime("%s"))

def train():
	csv = kaggle.file_to_array(training, 'all')
	times = query_times(csv, 4)
	all_skus = skus(csv)
	xtrees = ExtraTreesClassifier(n_estimators=1, max_depth=None, min_samples_split=1, random_state=0)
	model = xtrees.fit(times, all_skus)
	model = ConfidenceDecorator(model, times, all_skus)
	return model

def test_data():
	# validation set
	csv = kaggle.file_to_array(training, 'all')
	_times = query_times(csv)
	_all_skus = skus(csv)
	sku_times = {}
	for i in range(len(_all_skus)):
		sku = _all_skus[i]
		t = array(_times[i])
		if sku in sku_times:
			sku_times[sku].append(t)
		else:
			sku_times[sku] = [t]
	return sku_times

def real_data():
	csv = kaggle.file_to_array(testing, 'all')
	times = query_times(csv, 3)
	return times

def real_preds(model,times):
	preds = []
	for t in times:
		pred = model.predict(t)[0]
		preds.append([pred])
	return preds

def test_predictions(model, test_data):
	correct = 0.
	total = 0.
	preds = []
	for sku,times in test_data.items():
		for t in times:
			total += 1.
			pred = model.predict(t)[0]
			preds.append([pred])
			if sku == pred:
				correct += 1.
	print correct/total
	return preds

def real_test():
	model = train()
	times = real_data()
	return real_preds(model, times)
