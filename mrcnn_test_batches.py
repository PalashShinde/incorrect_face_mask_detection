from mrcnn.config import Config
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils , constants
import argparse
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from timeit import default_timer as timer
import csv
import json
import itertools as it
import pandas as pd
import df_confusion_matric
from datetime import date, datetime
import cv2
from keras.utils.vis_utils import plot_model
import shutil
import pathlib
import itertools  
from collections import defaultdict

POST_NMS_ROIS_INFERENCE = Config.POST_NMS_ROIS_INFERENCE
slidewise_threshold = df_confusion_matric.slidewise_threshold
bacteria_slidewise_threshold = slidewise_threshold['BACTERIA']
fungus_slidewise_threshold = slidewise_threshold['FUNGUS']
yeast_slidewise_threshold = slidewise_threshold['YEAST']

# csv_columns = ['Slide','Pathogen','TP_Act_P_Pred_P', 'TN_Act_N_Pred_N' ,'FP_Act_N_Pred_P','FN_Act_P_Pred_N' 
#                , 'Precision', 'Recall',  'Accuracy' , 'F1','Specificity'  , 'Negative_prediction' ,  'Analyzed_time' ,  'total_time_for_predictions', 'weight_model_name'
#                , 'POST_NMS_ROIS_INFERENCE' , 'slidewise_threshold' , 'prec_0.5-0.6,TP1' , 'prec_0.6-0.7,TP2' , 'prec_0.7-0.8,TP3' , 'prec_0.8-0.9,TP4' , 'prec_0.9-1,TP5' ] 

csv_columns = ['Slide','Pathogen','TP_Act_P_Pred_P', 'TN_Act_N_Pred_N' ,'FP_Act_N_Pred_P','FN_Act_P_Pred_N' , 'Remaining_TN', 'not_eligible_imgs'
               , 'Precision', 'Recall',  'Accuracy' , 'F1','Specificity'  , 'Negative_prediction' ,  'Analyzed_time' ,  'total_time_for_predictions', 'weight_model_name'
               , 'POST_NMS_ROIS_INFERENCE' , 'slidewise_threshold' , 'prec_0.1-0.2,TP1','prec_0.2-0.3,TP2','prec_0.3-0.4,TP3','prec_0.4-0.5,TP4','prec_0.5-0.6,TP5','prec_0.6-0.7,TP6', 'prec_0.7-0.8,TP7' , 'prec_0.8-0.9,TP8' , 'prec_0.9-1,TP9' ] 

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = constants.batches 
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    NUM_CLASSES = 1 + 4

class FiveConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = constants.batches 
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    NUM_CLASSES = 1 + 5

class OneConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = constants.batches 
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    NUM_CLASSES = 1 + 1

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

def make_folders_path():
    old_core_os_path = os.path.realpath(__file__) + '/'
    old_current_os_path = os.path.normpath( old_core_os_path + os.sep + os.pardir) + '/'
    pre_current_os_path = os.path.normpath( old_current_os_path + os.sep + os.pardir) + '/'
    os.makedirs(pre_current_os_path+ '/predicted_imgs', exist_ok = True )
    current_os_path = pre_current_os_path + 'predicted_imgs/'
    os.makedirs(current_os_path+ '/csv_files', exist_ok = True )
    csv_write_path = current_os_path + 'csv_files/'
    return current_os_path,csv_write_path

def new_grouper(with_path , n ):
    return [list(with_path[i:i+n]) for i in range(0,len(with_path),n)]

def top_n_results(df, count):
    top10_df = df[df.score > 0.5]
    top_results = top10_df.groupby(["class_id"]).apply(
        lambda x: x.sort_values(["score"], ascending=False)).reset_index(drop=True)
    return top_results.groupby(["class_id"]).head(count)

def process_eligibility(image):
    if image is None:
        return 0
    if image.size == 0:
        return 0
    else:
        total_sum = np.sum(np.logical_and.reduce(
            (image[:, :, 0] < 170, image[:, :, 1] < 170, image[:, :, 2] < 170)))
        return total_sum

def write_bbox(Bbox , image , name_file , class_ids_old ,scores ):
    y1, x1, y2, x2 =  Bbox
    rect_img = cv2.rectangle(image,(x1 ,y1),(x2, y2),(0,0,0),3)
    cv2.imwrite('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/pred_bbox/base_{0}_{1}_{2}.png'.format(name_file,class_ids_old,scores), rect_img )

def unique_range_scores_old(csv_path , pathogen):
    df = pd.read_csv(csv_path)
    True_df = df[ (df['PATHOGENS_x']==pathogen) & (df['PATHOGENS_y']==pathogen) ]
    scores_list = True_df['score'].tolist()
    str_scores = [ str(d) for d in scores_list ]
    range_scores_values = [ i[:3] for i in str_scores]
    unique_range = np.unique( range_scores_values , return_counts = True)
    return unique_range

def precision_calculations(all_df, lower=None, higher=None):
    return len(all_df[ all_df['score'].between(lower, higher, inclusive = True) ])

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

def unique_range_scores(csv_path , pathogen , others):
    df = pd.read_csv(csv_path)
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

def common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame, not_eligible,pathogen=None,others=None):
    data_n_svs = pd.read_csv(csv_path)
    # x is data_n_svs , y is pred_data
    final_df = df_confusion_matric.check_pathogens(pred_data, data_n_svs)
    # confusion_mattrix 
    confunsion_dict,stain = df_confusion_matric.creation_of_confusion_matric(final_df, stain= pathogen)
    # metrics values
    Extra_TN = any(final_df['PATHOGENS_y'].isin([pathogen,'EMPTY','JUNK'])) and final_df[ (final_df['PATHOGENS_x'] != pathogen) 
                                                                                & (~final_df['PATHOGENS_y'].isin([pathogen,'EMPTY','JUNK'])) ].dropna(axis = 0, how ='any')
    print('Total_TN', len(Extra_TN))
    print('not_eligible', len(not_eligible))
    Total_TN = len(Extra_TN['image_name']) + len(not_eligible)
    all_val_cal_dict = df_confusion_matric.calculate_all_values(confunsion_dict,Total_TN)
    # all_val_cal_dict = df_confusion_matric.calculate_all_values(confunsion_dict)
    # convert to csv file
    csv_path = pred_csvs_path+'pred_csvs/{0}_{1}.csv'.format(basename,weight_model_name)
    final_df.to_csv(csv_path)
    prec_tp = unique_range_scores(csv_path,pathogen,others)
    final_df = final_df.dropna(axis = 0, how ='any')
    # This TN includes for he in PATHOGENS_x all values other than bactetia and PATHOGENS_y all values other than bactetia ,'EMPTY','JUNK'
    
    with open(pred_csvs_path+'pred_csvs/final_dict_16.csv', 'a+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns , extrasaction='ignore')
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow({'Slide':basename,'Pathogen':stain,
            'TP_Act_P_Pred_P':  confunsion_dict['TP'], 
            'TN_Act_N_Pred_N':  confunsion_dict['TN'],
            'FP_Act_N_Pred_P': confunsion_dict['FP'],
            'FN_Act_P_Pred_N': confunsion_dict['FN'],
            'Remaining_TN':len(Extra_TN['image_name']),
            'not_eligible_imgs':len(not_eligible),
            'Precision':all_val_cal_dict['Precision'] ,
            'Recall':all_val_cal_dict['Recall'], 
            'Accuracy':all_val_cal_dict['Accuracy'], 
            'F1':all_val_cal_dict['F1'],
            'Specificity': all_val_cal_dict['Specificity'] , 
            'Negative_prediction': all_val_cal_dict['Negative_Prediction'],
            'Analyzed_time': total_time_frame,
            'total_time_for_predictions':total_time_for_predictions,
            'weight_model_name': weight_model_name,
            'POST_NMS_ROIS_INFERENCE':POST_NMS_ROIS_INFERENCE,
            'slidewise_threshold': fungus_slidewise_threshold ,
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
# csv_columns = ['Slide','Pathogen','TP_Act_P_Pred_P', 'TN_Act_N_Pred_N' ,'FP_Act_N_Pred_P','FN_Act_P_Pred_N' , 'Remaining_TN', 'not_eligible_imgs'
# , 'Precision', 'Recall',  'Accuracy' , 'F1','Specificity'  , 'Negative_prediction' ,  'Analyzed_time' ,  'total_time_for_predictions', 'weight_model_name'
# , 'POST_NMS_ROIS_INFERENCE' , 'slidewise_threshold' , 'prec_0.1-0.2,TP1','prec_0.2-0.3,TP1','prec_0.3-0.4,TP1','prec_0.4-0.5,TP1','prec_0.5-0.6,TP1','prec_0.6-0.7,TP1', 'prec_0.7-0.8,TP1' , 'prec_0.8-0.9,TP4' , 'prec_0.9-1,TP5' ]  

def mrcnn_prediction(file_names,dir_paths,base_svs_name,total_pngs,config,model,pathogens_list,weight_model_name):
    print("Prediction per Image")
    print("pathogens_list_current" , pathogens_list)
    print("NUM_CLASSES",config.NUM_CLASSES)
    df = pd.DataFrame(columns=['image_name', 'class_id', 'score'])
    final_list = []
    class_id = []
    score = []
    for i in file_names:
        name_file = i
        # [os.path.join(dir_paths, j) for j in file_names]
        image = cv2.imread(os.path.join(dir_paths, name_file))
        process_eligibility_img = process_eligibility(image)
        # if process_eligibility_img <= 500:
        #     not_eligible.append(name_file)
        if process_eligibility_img > 500:
            name_file_new = name_file
            # with open('/home/kalpit/Desktop/abc.txt', 'a+') as txt_file:
            #     txt_file.write( str(name_file_new) )
            #     txt_file.write("\n")
            start = timer()
            results = model.detect([image], verbose=1)
            end = timer()
            predictions_per_img = end-start
            # with open('/home/kalpit/Desktop/mrcnn_prediction.txt', 'a+') as txt_file:
            #     txt_file.write( str('mrcnn_prediction') )
            #     txt_file.write("\n")
            #     txt_file.write( str(name_file_new) )
            #     txt_file.write("\n")
            #     txt_file.write( str(predictions_per_img) )
            #     txt_file.write("\n")
            r = results[0]
            class_ids_old = r['class_ids']
            scores = r['scores']
            masks = r['masks']
            # masks = masks * 255
            # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            Bbox = r['rois']
            # print('Bbox:::', Bbox)
            # print('mask:::',masks.shape)
            # cv2.imwrite('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/pred_masks/dict_{0}_{1}_{2}.png'.format(name_file,class_ids_old,scores), np.float32(masks) )
            # Bbox:::: [[ 33   0 512 512]]
            # if masks is None:
            #     pass
            # else:
                # masks = masks * 255
                # if masks.shape != (512,512,1):
                #     masks =  masks + masks
                # print('masks::', masks.shape)
                # cv2.imwrite('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/pred_masks/dict_{0}_{1}_{2}.png'.format(name_file,class_ids_old,scores), np.float32(masks[:,:,0]) )
            # if np.any(masks) ==  False:
            #     pass
            # else:
            #     cv2.imwrite('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/pred_masks/dict_{0}_{1}_{2}.png'.format(name_file,class_ids_old,scores), masks )
            Bbox_lists = []
            if np.any(Bbox) == False :
                pass  
            else:
                Bbox_lists.append(Bbox.tolist())
            if len(class_ids_old) and len(Bbox_lists):
                # print('Bbox_lists', Bbox_lists)
                class_ids_old = [pathogens_list[x] for x in class_ids_old]
                non_junk_bbox = defaultdict(list)
                for classes,bboxes in zip(class_ids_old,Bbox_lists[0]):
                    non_junk_bbox[classes].append(bboxes)
                # new_junk_bbox.append(non_junk_bbox)
                # if 'junk' in non_junk_bbox.keys():
                #     del non_junk_bbox['junk']
                new_scores = scores
                per_class_area = {}
                per_class_bbox_count = {}
                for Bkeys in non_junk_bbox.keys():
                    per_class_bbox_count.update({Bkeys:len(non_junk_bbox[Bkeys])})
                    # area_bbox = sum([(Bbox[2]-Bbox[0])*(Bbox[3]-Bbox[1]) for Bbox in non_junk_bbox[Bkeys]])
                    per_class_area.update({Bkeys:sum([(Bbox[2]-Bbox[0])*(Bbox[3]-Bbox[1]) for Bbox in non_junk_bbox[Bkeys]])})
                area_bbox_sum = sum(per_class_area.values()) 
            else:
                class_ids_old = ['Empty']
                # scores = [0.5]
                new_scores = [0.5]
                area_bbox_sum = "None"
                per_class_area = {'Empty':0}
                per_class_bbox_count = {'Empty':1}

            print('file_name', name_file_new)
            print('class_ids_old', class_ids_old)
            # print('new_scores', new_scores)
            for score, class_id in zip(new_scores, class_ids_old):
                final_list.append({
                    "image_name": name_file,
                    "class_id": class_id,
                    "score": round(float(score),3),
                    'Area':per_class_area[class_id]/per_class_bbox_count[class_id],
                    'patho_count':per_class_bbox_count[class_id]
                })
            # print('final_list',final_list)
            # for score, class_id in zip(scores, class_ids_old):
            #     final_list.append({
            #         "image_name": name_file,
            #         "class_id": class_id,
            #         "score": float(score),
            #         'Area':area_bbox_sum
            #     })

    df = pd.DataFrame(final_list)
    # df.to_csv(pred_csvs_path+'pred_csvs/{0}_{1}_y_df.csv'.format(base_svs_name,weight_model_name))
    # df_sorted = df.sort_values('score', ascending=False).drop_duplicates(['image_name'])
    df = df.sort_values('score').drop_duplicates(subset=['image_name', 'class_id'], keep='last') 
    # df_sorted.to_csv('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/d_csvs/sorted_{}_{}.csv'.format(base_svs_name,weight_model_name))
    # df.to_csv('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/d_csvs/df_{}_{}.csv'.format(base_svs_name,weight_model_name))
    df.to_csv(pred_csvs_path+'pred_csvs/{0}_{1}_y_df.csv'.format(base_svs_name,weight_model_name))
    # df.to_csv('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/pred_masks/{0}_{1}_y_df.csv'.format(base_svs_name,weight_model_name))
    # top_count_df = top_n_results(df, count=2000)
    #Write predicted images on path
    # current_os_path,csv_write_path = make_folders_path()
    # print('current_os_path', current_os_path)
    # print('csv_write_path', csv_write_path)
    # weight_name_dir = os.path.basename(weight_model_name)
    # # make_csv_pred_files(df,  dir_paths , current_os_path , base_svs_name, csv_write_path,  patho='bacteria')
    # pred_dir_path = make_csv_pred_files(df, dir_paths , current_os_path , base_svs_name, csv_write_path, weight_name_dir, patho='yeast')
    # make_csv_pred_files(df, dir_paths , current_os_path , base_svs_name, csv_write_path, patho='fungus')
    # make_csv_pred_files(df, dir_paths , current_os_path , base_svs_name, csv_write_path, patho='junk')
    top_patho = defaultdict(list)
    # df1 = df.loc[(df['patho_count'] > 1)][['patho_count','image_name']]
    df1 = df[df.class_id != 'Empty'][['patho_count','image_name']]
    for i,j in zip(df1['patho_count'],df1['image_name']):
        top_patho[i].append(j)
    pred_dir_path = ''
    pred_files = df.image_name.tolist()
    not_eligible = list(set(file_names) - set(pred_files))
    # print('not_eligible',not_eligible)
    return df, top_patho, base_svs_name , pred_dir_path , not_eligible

def make_csv_pred_files(df, dir_paths , current_os_path , base_svs_name, csv_write_path ,weight_name_dir, patho=None):
    # Write predicted images on path
    df.to_csv( csv_write_path +'{0}_dict.csv'.format(base_svs_name))
    # pathogens_list = ['unused', 'bacteria', 'yeast', 'patho', 'junk']
    patho_df = df[df['class_id']== patho]
    patho_list = patho_df['image_name'].tolist()
    patho_path = [os.path.join(dir_paths, i) for i in patho_list ]
    os.makedirs( current_os_path + '/{0}/{1}/{2}'.format(weight_name_dir,patho,base_svs_name), exist_ok = True )
    for files in patho_path:
        shutil.copy(files , current_os_path + '/{0}/{1}/{2}'.format(weight_name_dir,patho,base_svs_name))
    return current_os_path + '/{0}/{1}/{2}'.format(weight_name_dir,patho,base_svs_name)

def predict_batches(file_names,dir_paths,base_svs_name,total_pngs,config,model,pathogens_list):
    print(" prediction on batches")
    print("pathogens_list_current" , pathogens_list)
    print("NUM_CLASSES",config.NUM_CLASSES)
    final_list = []
    with_path = [ os.path.join(dir_paths, j )  for j in file_names ]
    final_with_path =  new_grouper( with_path , config.BATCH_SIZE   )
    # print(final_with_path)
    for i in final_with_path:
        final_img_array = [cv2.imread(j) for j in i]
        img_list_path = [ j for j in i ] 
        if len(final_img_array) != (config.IMAGES_PER_GPU * config.GPU_COUNT):
            final_img_array = final_img_array + [final_img_array[0] for _ in range( (config.IMAGES_PER_GPU * config.GPU_COUNT) - len(final_img_array))]
        suffix_name = list(map(os.path.split , img_list_path))
        start = timer()
        results = model.detect_batches( final_img_array, verbose=1)
        end = timer()
        predictions_per_img = end-start
        # with open('/home/kalpit/Desktop/predict_batches.txt', 'a+') as txt_file:
        #     txt_file.write("\n")
        #     txt_file.write( str('predict_batches') )
        #     txt_file.write("\n")
        #     txt_file.write( str(suffix_name) )
        #     txt_file.write("\n")
        #     txt_file.write( str(predictions_per_img) )
        #     txt_file.write("\n")
        for i , j in zip(results, suffix_name):
            r = i
            class_ids_old = r['class_ids']
            scores = r['scores']
            if len(class_ids_old):
                class_ids_old = [pathogens_list[x]
                                    for x in class_ids_old]
            else:
                class_ids_old = ['Empty']
                scores = [0.5]
            print('file_name', suffix_name)
            print('class_ids_old', class_ids_old)
            for score, class_id in zip(scores, class_ids_old):
                final_list.append({
                    "image_name": j[1],
                    "class_id": class_id,
                    "score": float(score),
                })
    df = pd.DataFrame(final_list)
    df.to_csv(pred_csvs_path+'pred_csvs/{0}_y_df.csv'.format(base_svs_name))
    # top_count_df = top_n_results(df, count=2000)
    top_count_df = ''

    return df, top_count_df, base_svs_name

def test_model(MODEL_DIR = "", COCO_MODEL_PATH = ""):

    assert MODEL_DIR is not ""
    assert COCO_MODEL_PATH is not ""

    
    IMAGE_DIR_List = list_sub_dir(test_dir)
    for dir_paths in IMAGE_DIR_List:
        # base_svs_name = os.path.basename(os.path.normpath(str.split( dir_paths , '/tiles')[0]))
        print('dir_paths', dir_paths)
        weight_model_name = COCO_MODEL_PATH.split('/')[-1]
        base_svs_name = dir_paths.split('/')[-4]
        # base_svs_name = dir_paths.split('/')[-2]
        print('dir_paths', dir_paths)
        file_names = os.listdir(dir_paths)
        total_pngs = len(file_names)
        start = timer() 
        df, top_patho, base_svs_name , pred_dir_path , not_eligible = mrcnn_prediction(file_names, dir_paths, base_svs_name, total_pngs,config,model,pathogens_list,weight_model_name)
        # df, top_count_df, base_svs_name = predict_batches(file_names, dir_paths, base_svs_name, total_pngs,config,model,pathogens_list)
        end = timer()
        total_time_for_predictions= end-start
        # with open('/home/kalpit/Desktop/mrcnn_prediction.txt', 'a') as txt_file:
        #     txt_file.write("\n")
        #     txt_file.write( str('Total_Time_for_predss'))
        #     txt_file.write("\n")
        #     txt_file.write( str(len(file_names) ))
        #     txt_file.write("\n")
        #     txt_file.write( str(total_time_for_predictions) )
        #     txt_file.write("\n")
        # with open('/home/kalpit/Desktop/predict_batches.txt', 'a') as txt_file:
        #     txt_file.write("\n")
        #     txt_file.write( str('Total_Time_for_predss'))
        #     txt_file.write("\n")
        #     txt_file.write( str(len(file_names) ))
        #     txt_file.write("\n")
        #     txt_file.write( str(total_time_for_predictions) )
        #     txt_file.write("\n")
        precison = args.precision
        if precison:
            csv_path = csv_dir+base_svs_name+'.csv'
            current_time = datetime.now()
            total_time_frame = str(current_time)
            pred_data = df
            pred_data['PATHOGENS'] = pred_data['class_id'].str.upper()
            basename = base_svs_name
            basename = 'data'+basename
            # annotated_gms_df = [ pd.read_csv(list(os.walk(annotated_x_path))[0][0]+i) for i in list(os.walk(annotated_x_path))[0][2] if i.endswith('-gms.csv')]
            print("Base case ID", basename)
            if 'gms' in os.path.basename(csv_path):
                print("calculating precision on csv {0} and caseid {1}".format(os.path.basename(csv_path),basename))
                common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,not_eligible,pathogen='FUNGUS',others='JUNK' )
            elif 'he' in os.path.basename(csv_path):
                print("calculating precision on csv {0} and caseid {1}".format(os.path.basename(csv_path),basename))
                common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,not_eligible,pathogen='BACTERIA',others='JUNK' )
            elif 'pas' in os.path.basename(csv_path):
                print("calculating precision on csv {0} and caseid {1}".format(os.path.basename(csv_path),basename))
                common_pred(pred_data,csv_path,total_time_for_predictions,weight_model_name,current_time,basename,total_time_frame,not_eligible,pathogen='YEAST',others='JUNK' )
        else:
            print('MRCNN model prediction df in dir ==',pred_csvs_path+'pred_csvs/' )

    return df , pred_dir_path

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on pathogens.')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights single .h5 file ")
    parser.add_argument('--models_path', required=False,
                        metavar="/path/to/weights.h5",
                        help=" pass a path of dir containing multiple weights file")
    parser.add_argument('--test_dir', required=False,
                        metavar="/path/to/weights.h5",
                        help=" pass a path of dir containing multiple test dirs")
    parser.add_argument('--logs', required=False,
                        metavar="/path/to/logs/",
                        help='log directory (default=logs/)')
    parser.add_argument('--predictions', required=True,
                        metavar="/path/to/logs/",
                        help='four_class or five_class or one_class to select from')
    parser.add_argument('--precision', required=False,
                        metavar="/path/to/logs/",
                        help='Yes or No to to select from')
    parser.add_argument('--csv_dir', required=False,
                        metavar="/path/to/logs/",
                        help='Pass Csv dir i.e x csv files')
    args = parser.parse_args()
    test_dir = path_clean_end(args.test_dir)
    pred_csvs_path = os.path.normpath( test_dir + os.sep + os.pardir) + '/'
    os.makedirs(pred_csvs_path+'pred_csvs',exist_ok=True)
    if args.csv_dir:
        csv_dir = path_clean_end(args.csv_dir)
    # print('args.precision111', args.precision)
    if args.models_path:
        print('Prediction on multiple weights')
        models_path = path_clean_end(args.models_path)
        #LOOP WEIGHTS
        model_files = [models_path + m for m in os.listdir(models_path) if '.h5' in m]
        for model_path in model_files:
            print('model_path::', model_path)
            df , pred_dir_path = test_model(MODEL_DIR = args.logs, COCO_MODEL_PATH = model_path)
            # print(df.columns)
        print("completed")
        # print('pred_dir_path', pred_dir_path)
    else:
        print('Prediction on single weights')
        COCO_MODEL_PATH = args.model
        # SINGLE WEIGHT FILE
        MODEL_DIR = args.logs
        df = test_model(MODEL_DIR = MODEL_DIR, COCO_MODEL_PATH = COCO_MODEL_PATH)
        # print(df.columns)
        print("Predicted csv's are in ==",pred_csvs_path+'pred_csvs/' )