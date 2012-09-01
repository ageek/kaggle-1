import re

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

def format_words(data):
	output = {}
	for k,v in data.items():
		output[k] = re.sub("[\"']", '', v)
	return output

def slice(data, index):
	output = []
	for d in data:
		output.append(d[index])
	return output

data = file_to_array("../data/train.csv")
words = file_to_hash("../data/train.csv", 0, 3)
print format_words(words)
