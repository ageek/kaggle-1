import sys
sys.path.append("../../")
import kaggle

class BestBuy():
	training = "../data/training.csv"
	class_labels_index = 1

	def skus(cls):
		class_labels = kaggle.slice(kaggle.file_to_array(csv_file), class_labels_index)
		return class_labels
