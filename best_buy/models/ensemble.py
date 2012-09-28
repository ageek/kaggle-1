import knn
import tf_idf
import sys
sys.path.append("../../")
import kaggle
import time
import decision_tree
import vote

training = "../data/train.csv"

#def merge_answers(knn_preds, tf_idf_preds):
#	out = []
#	for i in range(len(knn_preds)):
#		sub_out = knn_preds[i]
#		for t_pred in tf_idf_preds[i]:
#			if str(t_pred) not in sub_out and len(sub_out) < 5:
#				sub_out.append(str(t_pred))
#		out.append(sub_out)
#	return out

def score(predictions, answers):
	sum_score = 0.
	for index,prediction in enumerate(predictions):
		correct_answer = answers[index]
		score = 0.
		if correct_answer in prediction:
			score = 1.0/(prediction.index(correct_answer)+1)
			sum_score += score
	predictions_count = float(len(predictions))
	score = sum_score/predictions_count
	return score

def validation_test(w):
	start = time.time()
	#answers = kaggle.slice(kaggle.file_to_array(training, True), 1)
	xtrees_preds, forest_preds, knn_preds = decision_tree.real_test()
	tf_preds = tf_idf.real_test(w)
	merged = vote.merge_answers(xtrees_preds, forest_preds, tf_preds, knn_preds)
	kaggle.write_predictions(merged, "../data/predictions_9_28_12.csv")
	#accuracy = score(merged, answers)
	print "Duration: " + str(time.time() - start)
	return None#accuracy

def real_test():
	start = time.time()
	knn_predictions = knn.real_test()
	print "\n\nStarting tf_idf\n\n"
	print "Duration: " + str(time.time() - start)
	tf_idf_predictions = tf_idf.real_test()
	merged = merge_answers(knn_predictions, tf_idf_predictions)
	kaggle.write_predictions(merged, "../data/predictions_9_26_12_b.csv")
	print "Duration: " + str(time.time() - start)
	return None

w = 1
print validation_test(w)
