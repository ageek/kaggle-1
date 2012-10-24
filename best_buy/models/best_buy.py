import sys
sys.path.append("../../")
import kaggle
import time
import re
from xml.dom.minidom import parseString

training = "/home/I829287/kaggle/best_buy/data/train.csv"
testing = "/home/I829287/kaggle/best_buy/data/test.csv"

def sku_to_searches():
	array = kaggle.file_to_array(training, "all")
	#array = array[0:10000:4] + array[1:10000:4] + array[2:10000:4]
	skus = kaggle.slice(array, 1)
	skus = set(skus)
	skus_searches = {}
	for sku in skus:
		skus_searches[sku] = []

	for line in array:
		sku = line[1]
		search = line[3]
		search = " ".join(kaggle.tokenize(search.lower()))
		search = re.sub("\"", '', search)
		skus_searches[sku].append(search)
	return skus_searches

def xml_to_array():
	output = []
	path = "../data/small_product_data.xml"

	file = open(path, "r")
	data = file.read()
	file.close()
	dom = parseString(data)
	products = dom.getElementsByTagName('product')
	for p in products:
		name = p.getElementsByTagName('name')[0].toxml()
		sku = p.getElementsByTagName('sku')[0].toxml()
		#description = p.getElementsByTagName('shortDescription')[0].toxml()
		temp = [sku, name]
		formatted = []
		for tag in temp:
			tag = tag.replace("<.*", '')
			tag = re.sub("<[\w\/]*>", '', tag)
			formatted.append(tag)
		output.append(formatted)
	return output

def csv_with_more_data():
	new_path = "../data/extra.csv"
	more_data = xml_to_array()
	old = kaggle.file_to_array(training)
	skus = set(kaggle.slice(old, 1))
	added_skus = set([])
	count = 0

	with open(new_path, "a") as new:
		for o in old:
			new.write(",".join(o).strip() + "\n")

		for product in more_data:
			sku = product[0]
			name = product[1]

			if sku in skus and sku not in added_skus:
				count += 1
				added_skus.add(sku)
				fake_line = ",".join(['fakeuser', sku, 'fakecategory', name, 'fake_time', 'fake_time']).encode('utf-8').strip()
				for i in range(15):
					new.write(fake_line + "\n")
	print len(added_skus)
	print len(skus)
	print count
	return None

#start = time.time()
#csv_with_more_data()
#print time.time() - start
