from collections import namedtuple
import pandas as pd
import csv

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

if __name__ == "__main__":

	dict_500_x = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/gms_870_bbox_data.csv').to_dict('records')
	dict_500_y = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/model_result_y.csv').to_dict('records')
	csv_columns = ['image_name','iou_score']
	for dict_x in dict_500_x:
		# {'x1': 3, 'x2': 166, 'y2': 187, 'y1': 4, 'image_name': '190226-3-261-gms-20_25_37_12800_18944_13312_19456.png', 'patho': 'fungus'}
		bbox_x = ( dict_x['x1'], dict_x['y1'],dict_x['x2'],dict_x['y2'] )
		bbox_x = list(map(float,bbox_x))
		# print('bbox_x',bbox_x)
		current_image = dict_x['image_name']
		for dict_y in dict_500_y:
			# {'class_id': 'FUNGUS', 'score': 59.687, 'Bbox': '(272, 352, 320, 400)', 'image_name': '190226-3-261-gms-20_10_40_5120_20480_5632_20992.png'}
			if dict_y['image_name'] == current_image and dict_y['score'] != 0 :
				bbox_y = ( dict_y['x1'], dict_y['y1'],dict_y['x2'],dict_y['y2'] )
				bbox_y = list(map(float,bbox_y))
				# print('dict_y',bbox_y)
				iou = bb_intersection_over_union(bbox_x, bbox_y)
				if iou != 0.0:
					print(current_image,iou)
					with open('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/iou_score.csv', 'a+') as csv_file:
						writer = csv.DictWriter(csv_file, fieldnames=csv_columns , extrasaction='ignore')
						if csv_file.tell() == 0:
							writer.writeheader()
						writer.writerow({'image_name':current_image,'iou_score':iou})