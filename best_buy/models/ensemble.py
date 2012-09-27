import knn
import tf_idf
import sys
sys.path.append("../../")
import kaggle
import time

def merge_answers(knn_preds, tf_idf_preds):
	out = []
	for i in range(len(knn_preds)):
		k_pred = knn_preds[i][0]
		sub_out = [str(int(k_pred))]
		for t_pred in tf_idf_preds[i]:
			if str(t_pred) not in sub_out and len(sub_out) < 5:
				sub_out.append(str(t_pred))
		out.append(sub_out)
	return out

start = time.time()
knn_predictions = knn.real_test()
print "\n\nStarting tf_idf\n\n"
print "Duration: " + str(time.time() - start)
tf_idf_predictions = tf_idf.real_test()
merged = merge_answers(knn_predictions, tf_idf_predictions)
kaggle.write_predictions(merged, "../data/predictions_9_26_12_b.csv")
print "Duration: " + str(time.time() - start)
