import knn
import tf_idf
import sys
sys.path.append("../../")
import kaggle

def merge_answers(knn_preds, tf_idf_preds):
	out = []
	for i in range(len(knn_preds)):
		k_preds = knn_preds[i][0]
		sub_out = [str(int(k_preds))]
		for t_pred in tf_idf_preds[i]:
			if str(t_pred) not in sub_out and len(sub_out) < 5:
				sub_out.append(str(t_pred))
		out.append(sub_out)
	return out

knn_predictions = knn.real_test()
kaggle.write_predictions(knn_predictions, "../data/predictions_9_11_12.csv")
#tf_idf_predictions = tf_idf.real_test()
#merged = merge_answers(knn_predictions, tf_idf_predictions)
#kaggle.write_predictions(merged, "../data/predictions_9_6_12.csv")
