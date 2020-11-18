import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import patches
import matplotlib.pyplot as plt 
import sys
import csv
import shutil
import pathlib

data_dir = '/home/k/Desktop/Work/ability/Faster_RCNN_2/'
train_img_path, mask_img_path = data_dir + 'images/', data_dir + 'masks/'


def extract_bboxes(mask, no_of_boxes = None):
   # Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    kernel = np.ones((31,31), np.uint8) 
    img_dilation = cv2.dilate(mask, kernel, iterations=1) 
    contours, hierarchy = cv2.findContours(img_dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    no_of_contours =  min(5, len(contours))
    boxes = np.zeros([len(contours), 4], dtype=np.int32)
    contours_area = []
    for i in range(len(contours)):
        z = np.zeros_like(mask, dtype = 'uint8')
        cv2.drawContours(z, contours[i], -1, (150, 255, 40), 3) 
        contours_area.append(cv2.contourArea(contours[i]))
        horizontal_indicies = np.where(np.any(z, axis=0))[0]
        vertical_indicies = np.where(np.any(z, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])

    area_idx = sorted(range(len(contours_area)), key=lambda i: contours_area[i], reverse=True) #[:no_of_boxes]
    final_bbox = np.array([boxes[k] for k in area_idx])

    return final_bbox.astype(np.int32)

def create_csv():


	print(len(os.listdir(train_img_path)))
	print(len(os.listdir(mask_img_path)))
	df = pd.DataFrame()
	x = {'1':'bacteria', '3':'fungus', '2':'yeast', '4':'junk'}
	df['images'] = os.listdir(train_img_path)
	df['class'] = df['images'].apply(lambda i:x[i[0]]) 
	df['image_files'] = [str(i) + '.png' for i in range(1, len(df)+1)]
	print(df)
	df['bbox'] = df['images'].apply(lambda x: extract_bboxes(cv2.imread(mask_img_path + x)[:,:,0])) 
	#df['no_of_bboxes'] = df['bbox'].apply(lambda x: len(x))
	#newdf = pd.DataFrame(np.repeat(df.values,np.max(df['no_of_bboxes']),axis=0))

	df1 = df.groupby(['images','class', 'image_files']).apply(lambda x: pd.DataFrame(x['bbox'].tolist()[0], columns=['y1','x1','y2','x2'])).reset_index([0,1,2]).reset_index(drop=True)
	df1 = df1.rename(columns={'class':'pathogen_type', 'image_files':'Image_names', 'y1':'ymin', 'x1':'xmin', 'y2':'ymax', 'x2': 'xmax'})
	df1.to_csv(data_dir+'new_data_dec_11.csv')

def load_csv(path):
    print(path)
    return pd.read_csv(path)

dir1 = '/home/konverge/officeproj/pathology/keras-frcnn/'
def visualize_bbox_image(img = '1-104_5_6_5109_6131_6133_7155_2.png'):
    df = load_csv(data_dir+'new_data.csv')
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    image = plt.imread(dir1+'train/' + img)
    plt.imshow(image)
    for _,row in df[df.images == img].iterrows():
        xmin = row.xmin
        xmax = row.xmax
        ymin = row.ymin
        ymax = row.ymax
        width = xmax - xmin
        height = ymax - ymin
        if row.pathogen_type == "bacteria":
            edgecolor = "r"
            ax.annotate('bacteria',xy=(xmax-40,ymin+20))

        elif row.pathogen_type == "fungus":
            edgecolor = "b"
            ax.annotate('fungus',xy=(xmax-40,ymin+20))

        elif row.pathogen_type == "yeast":
            edgecolor = 'g'
            ax.annotate('yeast',xy=(xmax-40,ymin+20))
        
        rect = patches.Rectangle((xmin,ymin),width,height,edgecolor=edgecolor,facecolor='none')
        ax.add_patch(rect)
    plt.show()

"""display input image with bboxes"""
# visualize_bbox_image()

def input_data_text_file():
    data = pd.DataFrame()
    df1 = load_csv(data_dir + 'new_data.csv')

    data['format'] = df1['images']
    # as the images are in train_images folder, add train_images before the image name
    # for i in range(data.shape[0]):
    for i in range(data.shape[0]):

        data['format'][i] = dir1 + 'train/' + data['format'][i]

    # add xmin, ymin, xmax, ymax and class as per the format required
    for i in range(data.shape[0]):
        data['format'][i] = data['format'][i] + ',' + str(df1['xmin'][i]) + ',' + str(df1['ymin'][i]) + ',' + str(df1['xmax'][i]) + ',' + str(df1['ymax'][i]) + ',' + df1['pathogen_type'][i]

    data.to_csv(dir1+'new_annotate.txt', header=None, index=None, sep=' ')

# input_data_text_file()
# vgg_data_to_text(data_dir+'new_data_dec_11.csv')
# vgg_data_to_text(data_dir+'data.csv')
data_frcnn_dir = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/data/training_data/'
def vgg_data_to_text_1(input_csv):
    data = pd.DataFrame()
    df1 = load_csv(input_csv)
    data['images'] = df1['images'].apply(lambda x:data_frcnn_dir+x)
    data['xmin'] = df1['xmin']
    data['ymin'] = df1['ymin']
    data['xmax'] = df1['xmax']
    data['ymax'] = df1['ymax']
    data['pathogen_type'] = df1['pathogen_type']
    data.to_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/data/' + 'new_data_dec_11_end.txt', header=None, index=None, sep=',')

####### Training Data Set Prep

def write_bbox(Bbox,image,img_name):
    # y1, x1, y2, x2 =  Bbox , 	(filename,class_name,y1,x1,y2,x2)
    y1 = Bbox['xmin']
    x1 = Bbox['xmax']
    y2 = Bbox['ymin']
    x2 = Bbox['ymax']
    # x1 = Bbox['xmin']
    # x2 = Bbox['xmax']
    # y1 = Bbox['ymin']
    # y2 = Bbox['ymax']
    rect_img = cv2.rectangle(image,(x1 ,y1),(x2, y2),(0,0,0),3)
    cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/completed_anno/base_{0}.png'.format(img_name),rect_img)

def read_csv_write_bbox(csv_path,train_dir):                        
	df = load_csv(csv_path)
	fileimgs_list = df.to_dict(orient='records')
	for files in fileimgs_list:
		image = cv2.imread(files['filename'])
		write_bbox(files,image,os.path.basename(files['filename']))

    # for img_name in os.listdir(train_dir):
    #     image = cv2.imread(train_dir+img_name)
    #     print(train_dir+img_name)
    #     df3 = df[df['Image_name'] == img_name]
    #     bbox_list = df3.to_dict(orient='records')
    #     for bbox in bbox_list:
    #         write_bbox(bbox,image,img_name)

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

def combine_csv(csv_path_1,csv_path_2,write_csv):
    df1 = load_csv(csv_path_1)    
    df2 = load_csv(csv_path_2)
    frames = [df1, df2]
    result = pd.concat(frames,sort=True)
    result.to_csv(write_csv)

def make_frcnn_traindir(input_csv_1,origin_folder_path,train_dir):
    fl = load_csv(input_csv_1)
    filename_list = fl['filename'].unique().tolist()
    print(len(filename_list))
    extension_list = ['.png']
    origin_folder_path = origin_folder_path
    dir_list = list_sub_dir(origin_folder_path,extension_list)
    for dir in dir_list:
        for imgs in os.listdir(dir):
            if imgs in filename_list:
                # print(dir+imgs)
                try:
                    shutil.copy(dir+imgs,train_dir+imgs)
                except shutil.SameFileError:
                    pass

def vgg_data_to_text(input_csv_2,write_txt,train_dir):
    data = pd.DataFrame()
    # df1 = pd.read_csv(input_csv_2,index_col=[0])
    df1 = load_csv(input_csv_2)
    # (filename,x1,y1,x2,y2,class_name) = line_split
    # data['pathogen_type'] = df1['region_attributes'].apply(eval).apply(lambda x:x['class'] if 'class' in x.keys() else "")
    data['filename'] = df1['filename'].apply(lambda x:train_dir+x)
    data['pathogen_type'] = df1['pathogen_type']
    data['xmin'] = df1['region_shape_attributes'].apply(eval).apply(lambda x: x["x"])
    data['ymin'] = df1['region_shape_attributes'].apply(eval).apply(lambda x: x["y"])
    data['xmax'] = df1['region_shape_attributes'].apply(eval).apply(lambda x: x["x"]+x["width"])
    data['ymax'] = df1['region_shape_attributes'].apply(eval).apply(lambda x: x["y"]+x["height"])
    # data['xmin'] = df1['xmin']
    # data['ymin'] = df1['ymin']
    # data['xmax'] = df1['xmax']
    # data['ymax'] = df1['ymax']
    print('csvssvss')
    data.set_index('filename', inplace=True)
    # data.to_csv(write_txt + 'newly_annotated_6march.txt', header=None, index=None, sep=str(','))
    data.to_csv(write_txt+'yeast_april21bbox_anno.csv')
    # return data

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, BboxParams
)

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility,label_fields=['category_id']))

def augmentattions():
    image = cv2.imread('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_he/190304-73-599-he-20_5_22_2560_11264_3072_11776.png')
    rows, cols = image.shape[:2]    
    img_h = cv2.flip(image, 1)
    img_v = cv2.flip(image, 0)
    bbox_list = [  {'xmin':10,'ymin':143,'xmax':178,'ymax':354} ]
    # write_bbox(Bbox,image,img_name)
    for bbox in bbox_list:
        write_bbox(bbox,image,'190304-73-599-he-20_5_22_2560_11264_3072_11776_before_h')
        xmin = bbox['xmin']
        xmax = bbox['xmax']
        bbox['xmax'] = cols - xmin
        bbox['xmin'] = cols - xmax
        write_bbox(bbox,img_h,'190304-73-599-he-20_5_22_2560_11264_3072_11776.png_after_h')
    print(bbox_list)
    # write_bbox(Bbox,image,img_name)
if __name__ == "__main__":
    ### Augmentations testing
    # image = cv2.imread('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_he/190304-73-599-he-20_5_22_2560_11264_3072_11776.png')
    # annotations = {'image': image, 'bboxes': [[10, 143, 178, 354]],'category_id': [1]}
    # # augmentation = strong_aug(p=0.9)
    # # augmented = augmentation(**data)
    # # w_img = augmented["image"]
    # category_id_to_name = {1: 'bacteria'}
    # aug = get_aug([HorizontalFlip(p=1)])
    # augmented = aug(**annotations)
    # img = augmented['image'].copy()
    # for idx, bbox in enumerate(annotations['bboxes']):
    #     x_min, y_min, w, h = bbox
    #     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    #     # write_bbox(bbox,img,'190304-73-599-he-20_5_22_2560_11264_3072_11776.png_after_albu_h')
    #     rect_img = cv2.rectangle(image,(x_min ,y_min),(x_max, y_max),(0,0,0),3)
    #     cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/completed_anno/base_albumtations.png',rect_img)
    #     # img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)

    ####################

    # df = load_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/top_20.csv')
    # image_list = os.listdir('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data_20')
    # fd = df[df['Image_name'].isin(image_list)]
    # print(len(fd['Image_name'].unique()))
    # fd.to_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/new_20_final.csv')

    #####

    # Prepare combine csv from tejas updated annotation
    # csv_path_1 = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/261-gms-20final.csv'
    # csv_path_2 = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/combined_239_241_237_6march.csv'
    # write_csv = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/combined_239_241_237_261_11march.csv'
    # combine_csv(csv_path_1,csv_path_2,write_csv)

    # # # Copy extra images from combined csv to train dir
    # origin_folder_path = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/data/'
    # input_csv_1 = csv_path_1
    train_dir = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_pas/pas_21_april/'
    # make_frcnn_traindir(input_csv_1,origin_folder_path,train_dir)

    # # # Prepare txt file for training from combined csv 
    # input_csv_2 = write_csv
    input_csv_2 = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/yet_to_combined/PAS_Combined.csv'
    write_txt = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/'
    vgg_data_to_text(input_csv_2,write_txt,train_dir)

    csv_file = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/yeast_april21bbox_anno.csv'
    txt_file = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/yeast_april_21.txt'
    with open(txt_file, "a+") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()

    ## Write training Bbox
    # csv_path = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/cvs_txt/bacteria_data_old.csv'
    # train_dir = '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_he/'
    # read_csv_write_bbox(csv_path,train_dir)

    
    
    
