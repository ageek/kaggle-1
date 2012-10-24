import knn
import sys
sys.path.append("../../")
import kaggle
#import vector

def adapt(correct, wrong):
	new = "../data/adapt_4.csv"

	with open(new, 'a') as new_file:
		for i,query in enumerate(correct):
			q = query[0]
			sku = query[1]
			output = ",".join(['x', sku, 'x', q, 'x', 'x']) + "\n"
			if i % 2 == 0:
				new_file.write(output)

		for i,query in enumerate(wrong):
			q = query[0]
			sku = query[1]
			output = ",".join(['x', sku, 'x', q, 'x', 'x']) + "\n"
			for x in range(2):
				new_file.write(output)

	return None
