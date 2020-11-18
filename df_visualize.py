import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import pathlib
import shutil
import random
import cv2
import shutil
import requests
from urllib import request, parse
import json
import string
import random
import csv

Precision_dict = {}
TP_dict = {}

csv_columns = ['description_5', 'image_3', 'pan', 'products', 'description', 'description_2', 'description_1', 'image_4'
, 'description_3', 'established_year', 'image_1', 'image_2', 'aadhar', 'gstn', 'user', 'distributor_id', 'contact_number', 
'profile_image', 'image_5', 'address', 'enrolled_shopkeepers', 'description_4', 'registered_name']

def csv_format():
    df = pd.read_excel('/home/kalpit/Encfs/work/User_Data.xlsx',sheet_name='Distributors')
    # print(df.columns)
    df1 = df.replace(np.nan, 'Not_Available', regex=True)
    # print(type(df.T.to_dict().values()))
    list_of_dict = df1.to_dict(orient='records')
    count = 0
    url = 'http://127.0.0.1:8000/vendor-service/v1/distributors/signup/'
    # print(i['Supplier'].split(' ')[0])
    # prin()
    for dict_json in list_of_dict:
        count +=1
        data ={
            "user": { 
                "username": dict_json['Supplier'].split(' ')[0]+ str(random.choice(string.ascii_uppercase).lower()),
                "password": "1234567890",
                "email": "testuser{0}@testdomain.com".format(random.choice(string.ascii_uppercase).lower())
            },
            "established_year": 1996+count,
            "registered_name": dict_json['Supplier'],
            "contact_number": 1234567890+count,
            "aadhar":  dict_json['PAN No.'],
            "pan":  dict_json['PAN No.'],
            "gstn":  dict_json['GSTIN'],
            "address": dict_json['Address']
        }
        # print('data',data)
        response = requests.post(url,json=data)
        print('r.status_code',response.status_code)
        r_json=response.json()
        print('response',r_json)

def dict_calulations(df,weights,patho=None):
    FUNGUS = df[ (df['weight_model_name']==weights) & (df['Pathogen']==patho)]
    TP_Act_P_Pred_P = FUNGUS.sum(axis = 0, skipna = True)['TP_Act_P_Pred_P']
    Precision = FUNGUS.mean(axis = 0)['Precision']
    Precision_dict.update({weights:Precision})
    TP_dict.update({weights:TP_Act_P_Pred_P})
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*sorted(dict.items())))
    ax.tick_params(axis ='x', rotation = -90)
    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/kalpit/Desktop/zero_files/books_read.png')
    # return Precision_dict,TP_dict

def plot_save(dict):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*sorted(dict.items())))
    ax.tick_params(axis ='x', rotation = -90)
    plt.tight_layout()
    plt.show()

def plot_df(path):
    df = pd.read_csv(path)
    weight_list = sorted(df['weight_model_name'].unique().tolist())
    for weights in weight_list:
        dict_calulations(df,weights,patho='FUNGUS')
        # plot_save(Precision_dict)
        # FUNGUS = df[ (df['weight_model_name']==weights) & (df['Pathogen']=='FUNGUS') ]
        # TP_Act_P_Pred_P = FUNGUS.sum(axis = 0, skipna = True)['TP_Act_P_Pred_P']
        # Precision = FUNGUS.mean(axis = 0)['Precision']
        # Precision_dict.update({weights:Precision})
        # TP_dict.update({weights:TP_Act_P_Pred_P})

def new_df(path):
    df = pd.read_csv(path)
    df1 = df[ (df['PATHOGENS_x']=='BACTERIA') & (df['PATHOGENS_y']=='BACTERIA')]
    # fig, ax = plt.subplots() 
    # print(df1['Area'].sort_values().value_counts(bins =50))
    # df['Area'].plot.line()
    # fig = plt.figure(figsize=(20,10))
    area = df1['Area'].sort_values()
    bins = list(range(int(area.min()), int(area.max())+1, 1000))
    # pd.cut(area, bins).cat.categories
    # df['Area'].sort_values().value_counts(bins =20).plot(kind='bar')
    df2 = pd.cut(area, bins).value_counts().to_frame()
    df3 = df2.sort_index(axis = 0) 
    print(df3)
    # print(df2.index.sort_values())
    # df2[''] = df2.apply(lambda x:x[0])
    # .plot(kind='bar')
    df3.plot(kind='bar', color = 'blue')
    plt.xlabel('Area Range')
    plt.ylabel('Frequency Count')
    plt.tight_layout()
    # for i, v in enumerate(df3['Area']):
    #     plt.text(v + 1, i + 1, str(v), color='blue', fontweight='bold')
    # df1['Area'].sort_values().plot.bar()
    
    plt.show()
    # plt.savefig('/home/kalpit/Desktop/zero_files/gms.png')

def del_junk_df(csv_list, g):
    for csv in csv_list:
        df = pd.read_csv(g+csv, index_col=[0])
        # del df['Unnamed: 0']
        df.columns.str.match('Unnamed')
        df = df.loc[:, ~df.columns.str.match('Unnamed')]
        # df.drop('Unnamed: 0', axis=1, inplace=True)
        # df = df[df.class_id != 'Empty']
        df.to_csv('/home/kalpit/Desktop/palash/projects/pathology/mrcnn_GPU/d_csvs/{}'.format(csv) , index=False)
    # if 'junk' in df.columns:
                #     del non_junk_bbox['junk']

def path_clean_end(origin_folder_path):
    true_path = origin_folder_path+'/' if not origin_folder_path.endswith('/') else origin_folder_path
    return true_path

def list_sub_dir(origin_folder_path,extension_list):
    new_folder_path = path_clean_end(origin_folder_path)
    root = pathlib.Path(new_folder_path)
    non_empty_dirs = {str(p.parent) for p in root.rglob('*') if p.is_file()}
    unique_list = []
    for dirs in non_empty_dirs:
        files = os.listdir(dirs)
        for pngs in files:
            for extns in extension_list:
                if pngs.endswith(extns):
                    unique_list.append(dirs)
                    break
    unique_list = set(unique_list)
    final_dir_list = sorted([ path_clean_end(paths) for paths in unique_list])
    return final_dir_list

# def list_sub_dir(origin_folder_path,extension_list):
#     new_folder_path = path_clean_end(origin_folder_path)
#     root = pathlib.Path(new_folder_path)
#     non_empty_dirs = {str(p.parent) for p in root.rglob('*') if p.is_file()}
#     unique_list = []
#     for dirs in non_empty_dirs:
#         files = os.listdir(dirs)
#         files_extension = set([ext.split('.')[1] for ext in files])
#         for pngs in files:
#             for extns in extension_list:
#                 if pngs.endswith(extns):
#                     unique_list.append(dirs)
#                     break
#     unique_list = set(unique_list)
#     final_dir_list = sorted([ path_clean_end(paths) for paths in unique_list])
#     return final_dir_list

def process_parent_folder_recursively(origin_folder_path):
    origin_folder_path = path_clean_end(origin_folder_path)
    origin_path_list = []
    max_sub_dir_count = max([len(x[0].split('/')) for x in os.walk(origin_folder_path)  ])
    origin_path_list = [x[0] for x in os.walk(origin_folder_path) if len(x[0].split('/')) == max_sub_dir_count]
    origin_path_list_new = [ path_clean_end(paths) for paths in origin_path_list ]
    return sorted (origin_path_list_new)


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        #ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=45, p=.2),
        #OneOf([
        #    IAAAdditiveGaussianNoise(),
        #    GaussNoise(),
        #], p=0.9),
        #OneOf([
        #    MotionBlur(p=.2),
        #    MedianBlur(blur_limit=3, p=.1),
        #    Blur(blur_limit=3, p=.1),
        #], p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2)
        #OneOf([
            # CLAHE(clip_limit=2),
        #    IAASharpen(),
        #    IAAEmboss(),
            # RandomContrast(),
            # RandomBrightness(),
        #], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p)

def classify_generator(folder_path):
    augmentation = strong_aug(p=0.9)
    class_dict = dict(enumerate(os.listdir(folder_path)))
    # class_dict {0: 'train_p', 1: 'train_n'}
    mappings = []
    for fol in class_dict.keys():
        for i in os.listdir(folder_path+class_dict[fol]):
            mappings.append((folder_path+class_dict[fol]+'/'+i, fol))

    for _ in range(100):

        x_train, y_train = [], []

        for s in random.sample(mappings, 1):
            print(s)
            resized = cv2.resize( cv2.imread(s[0])[:,:,0] , (128,128)) 

            data = {"image": np.expand_dims(resized, -1)}

            augmented = augmentation(**data)

            x_train.append(augmented["image"])
            # s = ('/datasets/train/train_p/00000708_000.png', 0) is [1.0, 0.0]
            # s = ('/datasets/train/train_n/00000708_000.png', 1) is [0.0, 1.0]
            # y_train.append([1.,0.] if not s[1] else [0.,1.])
            y_train.append(s[1])
            # print('x_train',x_train)
            # print('y_train',y_train)

        x, y =  np.array(x_train), np.array(y_train)
        yield x.shape, y


def count_csvs(g):
    csv_list = os.listdir(g)
    for csvs in csv_list:
        # print(csvs)
        if 'gms' in csvs:
            df = pd.read_csv(g+csvs)
            # print(csvs,len(df['image_name']))
            print(csvs,len(df[df['PATHOGENS'] == 'FUNGUS']))
        elif 'pas' in csvs:
            df = pd.read_csv(g+csvs)
            # print(csvs,len(df['image_name']))
            print(csvs,len(df[df['PATHOGENS'] == 'YEAST']))
        elif 'he' in csvs:
            df = pd.read_csv(g+csvs)
            # print(csvs,len(df['image_name']))
            print(csvs,len(df[df['PATHOGENS'] == 'BACTERIA']))

def union_count(normal_df,flip_df,x,y):
    TP_normal = normal_df[ (normal_df['Img_X'] == x) & (normal_df['Pred_Y'] == y)]
    TP_normal_list = TP_normal['Image_Name'].tolist()
    TP_flip = flip_df[ (flip_df['Img_X'] == x) & (flip_df['Pred_Y'] == y)]
    TP_flip_list = TP_flip['Image_Name'].tolist()
    # print(len(set(TP_normal_list)))
    # print(len(set(TP_flip_list)))
    print(len(set(TP_normal_list).union(set(TP_flip_list))))

def load_csv(path):
    print(path)
    return pd.read_csv(path)

def make_frcnn_traindir(input_csv,origin_folder_path,target_dir):
    fl = load_csv(input_csv)
    filename_list = fl['filename'].unique().tolist()
    print(len(filename_list))
    extension_list = ['.png']
    origin_folder_path = origin_folder_path
    dir_list = list_sub_dir(origin_folder_path,extension_list)
    for dir in dir_list:
        for imgs in os.listdir(dir):
            if imgs in filename_list:
                print(dir+imgs)
                try:
                    shutil.copy(dir+imgs,target_dir+imgs)
                except shutil.SameFileError:
                    pass

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
    # path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/'
    # extension_list = [ '.png','.jpeg']
    # df = pd.read_csv('/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/stage_2_detailed_class_info.csv')
    # nodes_list = [ i+'.jpeg' for i in df[ df['class']== 'No Lung Opacity / Not Normal']['patientId'].tolist() ]
    # print(nodes_list)
    # des_path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/nodes_data/'

    # import os

    # extensions = set()
    # # my_root = "./"  # some dir to start in

    # for root, dirs, files in os.walk(path) :
    #     for file in files: 
    #         pathname, exten = os.path.splitext(file) 
    #         extensions.add(exten)
    # print(extensions)
    # csv_list = sorted(os.listdir(path))
    # plot_df(path)
    # plot_save(Precision_dict)
    # new_df(path)
    # count_csvs(path)
    # h = list_sub_dir(path,extension_list)
    # for dir in h:
    #     for imgs in os.listdir(dir):
    #         if imgs in nodes_list:
    #             print(dir+imgs)
    #             shutil.move(dir+imgs,des_path)
    # folder_path  = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/train/'
    # train_gen = classify_generator(folder_path)
    # # next(train_gen)
    # for i,y in train_gen:
    #     print('1st',i)
    #     print('2nd',y)
    # print(final_dict)
    # print(sorted(final_dict.keys()))
    # del_junk_df(csv_list, path)
    # normal_df = pd.read_csv('/home/kalpit/User/palash_konverge/kaggle-data/512_I/pred_csv/normal_xy.csv')
    # flip_df = pd.read_csv('/home/kalpit/User/palash_konverge/kaggle-data/512_I/pred_csv/flip_xy.csv')
    # union_count(normal_df,flip_df,'Pneumonia','Normal')
    # dir_path = '/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/test_val/'

    # dir_list = process_parent_folder_recursively(dir_path)
    # # print(dir_list)
    # for paths in dir_list:
    #     # print(paths)
    #     if paths.endswith('Normal/') or paths.endswith('val_n/'):
    #         # print('jjjj')
    #         for imgs in os.listdir(paths):
    #             print(imgs)
    #             shutil.move(paths+imgs,'/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/train/train_n/'+imgs)
    #     elif paths.endswith('Pneumonia/') or paths.endswith('val_p/'):
    #         # print('kkk')
    #         for imgs in os.listdir(paths):
    #             print(imgs)
    #             shutil.move(paths+imgs,'/home/kalpit/User/palash_konverge/kaggle-data/512_I/extra_data/datasets/train/train_p/'+imgs)
    # User_test()
    # csv_format()
    ###########################

    # df3 = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/data/new.csv')
    # file_list = result['filename'].unique().tolist()
    # print(len(file_list))
    # 
    # print(dir_list)
    # for dir in dir_list:
    # dir = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/'
   
                
    # print(result.columns)
    # result.to_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/data/new.csv')
    # print(len(df3))418	36	450	73

    # image = cv2.imread('/home/kalpit/User/zero_files/cleaned/190312-20-239-gms-20_17_68_8704_34816_9216_35328.png')
    # rect_img = cv2.rectangle(image,(418,36),(450 ,73),(0,0,0),3)
    # cv2.imwrite('/home/kalpit/User/zero_files/cleaned/base1.png', rect_img )

    # df1 = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/img_anno.csv')
    # df2 = df1[['xmin','xmax','ymin','ymax']]
    # bbox_list = df2.to_dict(orient='records')

    # csv_path = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/data/combined_239_241_5march.csv'
    # train_dir = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/'
    # read_csv_write_bbox(csv_path,train_dir)

    # csv_file = '/home/kalpit/User/zero_files/newly_annotated_march.txt'
    # txt_file = '/home/kalpit/User/zero_files/new_text.txt'
    # with open(txt_file, "w") as my_output_file:
    #     with open(csv_file, "r") as my_input_file:
    #         [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    #     my_output_file.close()
    # import shutil
    # df = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/mrcnn_data/data/final_csv/190111-10-319-gms-20.csv')
    # junk_image_name = df[df['PATHOGENS'] == 'JUNK']['name_file']

    # try:
    #     for i in junk_image_name:
    #         print(i)
    #         shutil.copy('/home/kalpit/User/palash_konverge/projects/ability/mrcnn_data/data/valid_data/190111-10-319-gms-20/tiles/15/'+i,'/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/dummy_test_normal/')
    # except FileNotFoundError:
    #     pass
    
    # df_x = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/gms_test_data_x.csv')
    # df_y = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/model_result.csv')
    # df3 = pd.merge(df_x,df_y, on=['image_name'])
    # df3.to_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/result_xy.csv')

	# from collections import namedtuple
	# dict_500_x = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/gms_870_bbox_data.csv').to_dict('records')
	# dict_500_y = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/output/model_result_y.csv').to_dict('records')
	# colunmns 
	# for dict_x in dict_500_x:
	# 	# {'x1': 3, 'x2': 166, 'y2': 187, 'y1': 4, 'image_name': '190226-3-261-gms-20_25_37_12800_18944_13312_19456.png', 'patho': 'fungus'}
	# 	bbox_x = ( dict_x['x1'], dict_x['y1'],dict_x['x2'],dict_x['y2'] )
	# 	bbox_x = list(map(float,bbox_x))
	# 	# print('bbox_x',bbox_x)
	# 	current_image = dict_x['image_name']
	# 	for dict_y in dict_500_y:
	# 		# {'class_id': 'FUNGUS', 'score': 59.687, 'Bbox': '(272, 352, 320, 400)', 'image_name': '190226-3-261-gms-20_10_40_5120_20480_5632_20992.png'}
	# 		if dict_y['image_name'] == current_image and dict_y['score'] != 0 :
	# 			bbox_y = ( dict_y['x1'], dict_y['y1'],dict_y['x2'],dict_y['y2'] )
	# 			bbox_y = list(map(float,bbox_y))
	# 			# print('dict_y',bbox_y)
	# 			iou = bb_intersection_over_union(bbox_x, bbox_y)
	# 			print(current_image,iou)
    # dir_path = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/input/test_dir_yeast/'
    # extension_list = [ '.png' ]
    # dir_list = list_sub_dir(dir_path,extension_list)
    # csv_path = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/metrics_dir/input/csvs/'
    # for dirs in dir_list:
    #     base_svs_name = dirs.split('/')[-4]
    #     # for _ in os.listdir(csv_path):
    #     df = pd.read_csv(csv_path+base_svs_name+'.csv')
    #     png_list = os.listdir(dirs)
    #     c_d_list = df['image_name'].unique().tolist()
        # print(base_svs_name,set(c_d_list).issubset(png_list))
        # print('current',base_svs_name,len(os.listdir(dirs)),len(df['image_name'].unique()))
    # data = df[(df['PATHOGENS_x'] == 'BACTERIA') & (df['PATHOGENS_y'] == 'BACTERIA')]
    df = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/final_combined/yeast_april21bbox_anno.csv')
    # for imgs in df['filename'].unique():
    #     shutil.copy('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/pas_data/299/'+imgs,
    #     '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/pas_data/299_gms/')
    print('current',len(df['filename'].unique()))



