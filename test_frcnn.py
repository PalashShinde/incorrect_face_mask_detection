from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import argparse
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer
from datetime import date, datetime
import pathlib
import df_confusion_matric
from collections import defaultdict
import csv

sys.setrecursionlimit(40000)

csv_columns = ['Slide','Pathogen','TP_Act_P_Pred_P', 'TN_Act_N_Pred_N' ,'FP_Act_N_Pred_P','FN_Act_P_Pred_N' 
,'Precision', 'Recall',  'Accuracy' , 'F1','Specificity'  , 'Negative_prediction' ,  'Analyzed_time' 
,  'total_time_for_predictions', 'weight_model_name', 'prec_0.1-0.2,TP1','prec_0.2-0.3,TP2','prec_0.3-0.4,TP3','prec_0.4-0.5,TP4','prec_0.5-0.6,TP5','prec_0.6-0.7,TP6', 'prec_0.7-0.8,TP7' , 'prec_0.8-0.9,TP8' , 'prec_0.9-1,TP9' ] 

def path_clean_end(origin_folder_path):
    true_path = origin_folder_path+'/' if not origin_folder_path.endswith('/') else origin_folder_path
    return true_path

def list_sub_dir(origin_folder_path):
    new_folder_path = path_clean_end(origin_folder_path)
    root = pathlib.Path(new_folder_path)
    non_empty_dirs = {str(p.parent) for p in root.rglob('*') if p.is_file()}
    unique_list = []
    for dirs in non_empty_dirs:
        files = os.listdir(dirs)
        for pngs in files:
            if pngs.endswith('.png'):
                unique_list.append(dirs)
                break
    unique_list = set(unique_list)
    final_dir_list = sorted([ path_clean_end(paths) for paths in unique_list])
    return final_dir_list

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]																																																																							
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))
	return (real_x1, real_y1, real_x2 ,real_y2)

def initiate_test_model(C,model_path):

	class_mapping = C.class_mapping
	print('class_mapping11',class_mapping)																																																																																																																																																																																																																																																																																																																																																																																																																																																																		
	# if 'bg' not in class_mapping:
	# 	class_mapping['bg'] = len(class_mapping)
	class_mapping = {v: k for k, v in class_mapping.items()}
	print('class_mapping22',class_mapping)
	C.num_rois = int(args.num_rois)
	# num_features = 512
	print('num_features',num_features)
	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (num_features, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, num_features)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)
	# define the base network (resnet here, can be VGG, Inception, etc)
	# shared_layers = nn.nn_base(img_input, trainable=True)
	# shared_layers = nn.resnet_graph(img_input, architecture='resnet101')[-1]
	shared_layers = nn.resnet_graph(img_input, architecture='resnet101')[-1]
	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)
	classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
	model_rpn = Model(img_input, rpn_layers)
	# model_classifier = Model([img_input, roi_input], classifier)
	model_classifier = Model([feature_map_input, roi_input], classifier)
	# model_path = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/Faster_RCNN_2/good_weights/saved-model-march19-130_1.8432133417942862.hdf5'
	# print('Loading weights from {}'.format(C.model_path))
	print('inlinne_model_path',model_path)
	model_rpn.load_weights(model_path, by_name=True)
	# print(model_rpn.summary())
	# print(model_classifier.summary())
	model_classifier.load_weights(model_path, by_name=True)
	# print('Loading weights from {}'.format(C.model_path))
	# model_rpn.load_weights(C.model_path, by_name=True)
	# model_classifier.load_weights(C.model_path, by_name=True)
	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	return class_mapping,model_rpn,model_classifier

def slide_to_csv(img_path,class_mapping,model_rpn,model_classifier):
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	bbox_threshold = 0.4
	df = pd.DataFrame()
	# image_name_list,class_y,score = [],[],[]
	final_list = []
	#os.mkdir(dir1+str(slide_num))
	# folders = [x for x in os.listdir(dir2) if os.path.isdir(dir2+x)]
	# final_folder = [f1 for f1 in folders if str(slide_num) in f1]
	# print(final_folder[0])
	# final_path = dir2 + final_folder[0] + '/tiles/20'
	# final_path = dir1
	# for img_path in testimgs_dir_list:
	for idx, img_name in enumerate(sorted(os.listdir(img_path))):
		if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
			continue
		# print(img_name)
		st = time.time()
		# print('img_name',img_name)
		filepath = os.path.join(img_path,img_name)
		img = cv2.imread(filepath)
		X, ratio = format_img(img, C)
		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))
		# get the feature maps and output from the RPN
		[Y1, Y2, F] = model_rpn.predict(X)
		# print('model_rpn_predict')
		# print('Y1',[Y1, Y2, F][0])
		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.9)
		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]
		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}
		for jk in range(R.shape[0]//C.num_rois + 1):
			ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break
			if jk == R.shape[0]//C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded
			[P_cls, P_regr] = model_classifier.predict([F, ROIs])
			# print('P_cls',P_cls.shape[1])
			# print('P_regr',P_regr)
			for ii in range(P_cls.shape[1]):
				if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					# print('P_cls',P_cls.shape)
					continue
				cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
				# print('cls_name2',cls_name)
				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []
				(x, y, w, h) = ROIs[0, ii, :]
				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))
		all_dets = []
		for key in bboxes:
			bbox = np.array(bboxes[key])		
			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.9)
			# print('new_boxes',new_boxes)
			# print('new_probs',new_probs)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]
				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
				cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
				# textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				textLabel = '{}: {}'.format(key,int(new_probs[jk]))
				all_dets.append((key,new_probs[jk],(real_x1, real_y1, real_x2, real_y2)))
				# all_dets.append((key,100*new_probs[jk]))
				(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				textOrg = (real_x1, real_y1-0)
				cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
				cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
		# print('Elapsed time = {}'.format(time.time() - st))
		# print('all_dets',all_dets)
		# cv2.imshow('img', img)
		if len(all_dets) == 0:
			# for class_id,score,bbox in all_dets:
			print(img_name,'JUNK')
			final_list.append({
				"image_name": img_name,
				"class_id": 'JUNK',
				"score": 0.,
				'Bbox':'No_bbox',
			})
		else:
			for class_id,scores,bbox in all_dets:			
				print(img_name,class_id,scores,bbox)
				# cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/model_result/{0}'.format(img_name),img)
				final_list.append({
					"image_name": img_name,
					"class_id": class_id.upper(),
					"score": round(scores,4),
					'Bbox':bbox,
				})
		# image_name_list.append(img_name)
		# if len(all_dets) == 0:
		# 	class_y.append('Not_Predicted')
		# 	score.append(0.)
		# else:
		# 	print('all_dets[0][0]',all_dets)
		# 	class_y.append(all_dets[0][0].upper())
		# 	score.append(all_dets[0][1])
		# 	print('all_dets[0][0]',all_dets[0][0].upper())

			# print("writing {0} with {1}".format(img_name, bboxes))
			# cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/model_result/{0}'.format(img_name),img)
	# df['image_name'] = image_name_list
	# df['PATHOGENS_y'] = class_y
	# df['score'] = score
	# df.to_csv(out_csv_path+'/model_result_y.csv')

	df = pd.DataFrame(final_list)
	df = df.sort_values('score').drop_duplicates(subset=['image_name', 'class_id','Bbox'], keep='last')
	# df = df.sort_values('score').drop_duplicates(subset=['image_name', 'class_id'], keep='last')
	# df.to_csv(out_csv_path+'/model_result_y.csv')
	return df

# slide_to_csv(img_path)
def total_prec_cal(TP_df, FP_df, lower=None, higher=None):
    TP = len(np.unique(TP_df[ (TP_df['score'] >= lower) & (TP_df['score'] < higher) ]['image_name']))
    # TP = len(np.unique(TP_df[ TP_df['score'].between(lower, higher, inclusive = True) ]['image_name']))
    # True_df[ (df['score'] >= 0.5) & (df['score'] < 0.6)  ]
    FP = len(np.unique(FP_df[ (FP_df['score'] >= lower) & (FP_df['score'] < higher) ]['image_name']))
    # FP = len(np.unique(FP_df[ FP_df['score'].between(lower, higher, inclusive = True) ]['image_name']))
    if TP or FP:
        try:
            if int(TP+FP) != 0:
                prec = int(TP)/int(TP+FP)
                return prec,TP
            else:
                return 0,0
        except ZeroDivisionError:
            pass
    else:
        # no_prec = "NonPrecValues"
        return "NonPrecValues",'No_TP'

def unique_range_scores(df , pathogen , others):
    # df = csv_path
    True_df = df[ (df['PATHOGENS_x']==pathogen) & (df['PATHOGENS_y']==pathogen) ]
    FP_df = df[ (df['PATHOGENS_x']==others) & (df['PATHOGENS_y']==pathogen) ]
    # [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    # group_range = [ d[i:i+2] for i in range(0,len([ round(i,2) for i in np.arange(0.1,1.1,0.1).tolist() ]),2)]
    range_list = [ round(i,2) for i in np.arange(0.1,1.1,0.1).tolist() ]
    group_range = list(map(tuple, zip(range_list, range_list[1:])))
    # prec_tp = {}
    prec_tp = defaultdict(list)
    for float_ranges in group_range:
        precs, TPs = total_prec_cal(True_df, FP_df ,lower=float_ranges[0], higher=float_ranges[1])
        prec_tp.update({float_ranges:[precs, TPs]})
    return prec_tp

def common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,pathogen=None,others=None):
	data_n_svs = pd.read_csv(csv_path)
	# x is data_n_svs , y is pred_data
	
	final_df = df_confusion_matric.check_pathogens(pred_data, data_n_svs)
	final_df = final_df.sort_values('score').drop_duplicates(subset=['image_name', 'PATHOGENS_x','PATHOGENS_y'], keep='last')
	# confusion_mattrix 
	final_out_csv = out_csv_path+'{0}_{1}.csv'.format(basename,weight_model_name)
    # final_df = final_df.sort_values('score').drop_duplicates(subset=['image_name', 'PATHOGENS_x','PATHOGENS_y'], keep='last')
	# final_out_csv = out_csv_path+'{0}_{1}.xlsx'.format(basename,weight_model_name)
	# print('final_out_csv22',final_out_csv)
	# writer_e = pd.ExcelWriter(final_out_csv, engine='xlsxwriter')
	# final_df.to_excel(writer_e, sheet_name='Total')
	final_df.to_csv(final_out_csv)
	confunsion_dict,stain = df_confusion_matric.creation_of_confusion_matric(final_df,final_out_csv,stain= pathogen)
	# metrics values
	# Extra_TN = any(final_df['PATHOGENS_y'].isin([pathogen,'EMPTY','JUNK'])) and final_df[ (final_df['PATHOGENS_x'] != pathogen) 
	#                                                                             & (~final_df['PATHOGENS_y'].isin([pathogen,'EMPTY','JUNK'])) ].dropna(axis = 0, how ='any')
	# print('Total_TN', len(Extra_TN))
	# Total_TN = len(Extra_TN['image_name']) + len(not_eligible)
	Total_TN = 0
	all_val_cal_dict = df_confusion_matric.calculate_all_values(confunsion_dict,Total_TN)
	# all_val_cal_dict = df_confusion_matric.calculate_all_values(confunsion_dict)
	# convert to csv file
	# final_df.to_csv(final_out_csv)
	prec_tp = unique_range_scores(final_df,pathogen,others)
	final_df = final_df.dropna(axis = 0, how ='any')
	# This TN includes for he in PATHOGENS_x all values other than bactetia and PATHOGENS_y all values other than bactetia ,'EMPTY','JUNK'
	with open(out_csv_path+'/frcnn_metrics.csv', 'a+') as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=csv_columns , extrasaction='ignore')
		if csv_file.tell() == 0:
			writer.writeheader()
		writer.writerow({'Slide':basename,'Pathogen':stain,
			'TP_Act_P_Pred_P':  confunsion_dict['TP'], 
			'TN_Act_N_Pred_N':  confunsion_dict['TN'],
			'FP_Act_N_Pred_P': confunsion_dict['FP'],
			'FN_Act_P_Pred_N': confunsion_dict['FN'],
			'Precision':all_val_cal_dict['Precision'] ,
			'Recall':all_val_cal_dict['Recall'], 
			'Accuracy':all_val_cal_dict['Accuracy'], 
			'F1':all_val_cal_dict['F1'],
			'Specificity': all_val_cal_dict['Specificity'] , 
			'Negative_prediction': all_val_cal_dict['Negative_Prediction'],
			'Analyzed_time': total_time_frame,
			'total_time_for_predictions':total_time_for_predictions,
			'weight_model_name': weight_model_name,
			'prec_0.1-0.2,TP1': prec_tp[(0.1, 0.2)],
			'prec_0.2-0.3,TP2': prec_tp[(0.2, 0.3)],
			'prec_0.3-0.4,TP3': prec_tp[(0.3, 0.4)],
			'prec_0.4-0.5,TP4': prec_tp[(0.4, 0.5)],
			'prec_0.5-0.6,TP5': prec_tp[(0.5, 0.6)]  , 
			'prec_0.6-0.7,TP6' : prec_tp[(0.6, 0.7)]  , 
			'prec_0.7-0.8,TP7' : prec_tp[(0.7, 0.8)], 
			'prec_0.8-0.9,TP8' : prec_tp[(0.8, 0.9)] , 
			'prec_0.9-1,TP9' : prec_tp[(0.9, 1)]
			})

def test_model(testimgs_dir_list,C,model_path):

    # IMAGE_DIR_List = testimgs_dir_list
	for dir_paths in testimgs_dir_list:
		# ## base_svs_name = os.path.basename(os.path.normpath(str.split( dir_paths , '/tiles')[0]))
		# print('dir_paths', dir_paths)
		# weight_model_name = model_path.split('/')[-1]
		# base_svs_name = dir_paths.split('/')[-4]
		# # base_svs_name = dir_paths.split('/')[-2]
		# print('dir_paths', dir_paths)
		# file_names = os.listdir(dir_paths)
		# # total_pngs = len(file_names)
		# start = timer() 
		# class_mapping,model_rpn,model_classifier = initiate_test_model(C,model_path)
		# print('starting_predictsionssssss')
		# df = slide_to_csv(dir_paths,class_mapping,model_rpn,model_classifier)
		# end = timer()
		# total_time_for_predictions= end-start
		precison = args.precision
		if precison == 'Yes':
			print('dir_paths', dir_paths)
			weight_model_name = model_path.split('/')[-1]
			base_svs_name = dir_paths.split('/')[-4]
			# base_svs_name = dir_paths.split('/')[-2]
			print('dir_paths', dir_paths)
			# total_pngs = len(file_names)
			start = timer() 
			class_mapping,model_rpn,model_classifier = initiate_test_model(C,model_path)
			print('starting_predictsionssssss')
			df = slide_to_csv(dir_paths,class_mapping,model_rpn,model_classifier)
			end = timer()
			total_time_for_predictions= end-start
			######
			csv_path = csv_dir+base_svs_name+'.csv'
			current_time = datetime.now()
			total_time_frame = str(current_time)
			pred_data = df
			pred_data['PATHOGENS_y'] = pred_data['class_id']
			basename = base_svs_name
			basename = 'data'+basename
			# not_eligible = 0
			# annotated_gms_df = [ pd.read_csv(list(os.walk(annotated_x_path))[0][0]+i) for i in list(os.walk(annotated_x_path))[0][2] if i.endswith('-gms.csv')]
			if 'gms' in os.path.basename(csv_path):
				print("calculating precision on csv {0} and caseid {1}".format(os.path.basename(csv_path),basename))
				common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,pathogen='FUNGUS',others='JUNK' )
			elif 'he' in os.path.basename(csv_path):
				print("calculating precision on csv {0} and caseid {1}".format(os.path.basename(csv_path),basename))
				common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,pathogen='BACTERIA',others='JUNK' )
			elif 'pas' in os.path.basename(csv_path):
				print("calculating precision on csv {0} and caseid {1}".format(os.path.basename(csv_path),basename))
				common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,pathogen='YEAST',others='JUNK' )
		elif precison == 'No':
			class_mapping,model_rpn,model_classifier = initiate_test_model(C,model_path)
			print('starting_predictsionssssss')
			df = slide_to_csv(dir_paths,class_mapping,model_rpn,model_classifier)
			df.to_csv(out_csv_path+'{0}_{1}.csv'.format(basename,weight_model_name))
			print('FRCNN model prediction y_df in dir ==',out_csv_path)
		else: 
			print('Select Precision as Yes or No')

if __name__ == "__main__":

	# parser1 = OptionParser()
	# parser1.add_option("-n", "--num_rois", type="int", dest="num_rois",
	# 				help="Number of ROIs per iteration. Higher means more memory use.", default=3)
	# parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
	# parser.add_option("--config_filename", dest="config_filename", help=
	# 				"Location to read the metadata related to the training (generated when training).",
	# 				default="config.pickle")
	# parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50,resnet101.", default='vgg')
	parser = argparse.ArgumentParser(description='Test Faster R-CNN on pathogens.')
	parser.add_argument('--config_filename', required=True,
                        metavar="config_filename.",
                        help="Location to read the metadata related to the training (generated when training). ")
	parser.add_argument('--model', required=False,
					metavar="/path/to/weights.h5",
					help="Path to weights single .h5 file ")
	parser.add_argument('--models_path', required=False,
					metavar="/path/to_dir_of/ weights.h5",
					help=" pass a path of dir containing multiple weights file")
	parser.add_argument('--test_dir', required=False,
					metavar="/path/to/test_imgs_root_dir",
					help=" pass a path of dir containing multiple test dirs")
	parser.add_argument('--network', required=False,
					metavar="network to select from",
					help='Base network to use. Supports vgg or resnet50,resnet101.',default='vgg')
	parser.add_argument('--precision', required=False,
                        metavar="enable metrics cal",
                        help='Yes or No to to select from')
	parser.add_argument("--num_rois",required=False,
                        metavar="num_rois",
                        help='num_rois',default=3)
	parser.add_argument('--csv_dir', required=False,
                        metavar="/path/to/logs/",
                        help='Pass Csv dir i.e x csv files')
	# (options1, args) = parser1.parse_args()
	args = parser.parse_args()
	# img_path = options.test_path
	test_dir = path_clean_end(args.test_dir)
	if not args.test_dir:   # if filename is not given
		parser.error('Error: path to test data must be specified. Pass --path to command line')
	config_output_filename = args.config_filename
	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)
	if args.csv_dir:
		csv_dir = path_clean_end(args.csv_dir)
	if C.network == 'resnet50':
		print('importing resnet50')
		import keras_frcnn.resnet as nn
		num_features = 1024
	elif C.network == 'resnet101':
		print('importing resnet101')
		import keras_frcnn.resnet as nn
		# num_features = 512
		num_features = 1024
	elif C.network == 'vgg':
		import keras_frcnn.vgg as nn
		num_features = 512
	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False
	C.transpose = False

	out_csv_path = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/'
	testimgs_dir_list = list_sub_dir(test_dir)
	if args.models_path:
		print('Prediction on multiple weights')
		models_path = path_clean_end(args.models_path)
		#LOOP WEIGHTS
		model_files = [models_path + m for m in os.listdir(models_path) if '.h5' or '.hdf5' in m]
		for model_path_w in model_files:
			print('model_path::', model_path_w)
			test_model(testimgs_dir_list,C,model_path_w)
			# print(df.columns)
		print("completed")
	else:
		print('Prediction on single weights')
		test_model(testimgs_dir_list,C,args.model)
	
    # NO Precision
	# model_path = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/Faster_RCNN_2/good_weights/saved-model-march19-130_1.84.hdf5'
	# img_path = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/'
	# class_mapping,model_rpn,model_classifier = initiate_test_model(C,model_path)
	# slide_to_csv(img_path,class_mapping,model_rpn,model_classifier)
	


	

	