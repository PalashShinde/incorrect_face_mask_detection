import cv2
import numpy as np
import copy
import random
from keras_frcnn import config 

bconfig = config.Config()

from albumentations import (
    HorizontalFlip, BboxParams ,Transpose ,Rotate,Compose,VerticalFlip,normalize_bbox
)

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 
                                               min_visibility=min_visibility,label_fields=['category_id']))

def apply_augmentations(img_data_aug,img,aug,class_m):
	bbox_after_aug = []
	for extra_key in img_data_aug['bboxes']:
		del extra_key['class']
	for bbox in img_data_aug['bboxes']:
		annotations = {'image': img, 'bboxes': [ [bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']] ] ,'category_id': [class_m]}
		augmented = aug(**annotations)
		x1,y1,x2,y2 = augmented['bboxes'][0] 
		bbox_after_aug.append({'x1':abs(int(x1)),'y1':abs(int(y1)),'x2':abs(int(x2)),'y2':abs(int(y2))})
	return bbox_after_aug ,augmented

def augment(img_data, config, class_mapping,augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)
	img_data_aug_ori = copy.deepcopy(img_data)
	# print('img_data_aug_ori before',img_data_aug_ori)
	img = cv2.imread(img_data_aug['filepath'])
	class_m = class_mapping['fungus']
	if augment:
		rows, cols = img.shape[:2]
		augment_list = [ 
		{'horizontal_flips':bconfig.use_horizontal_flips} 
		,{'vertical_flips':bconfig.use_vertical_flips} 
		,{'transpose':bconfig.transpose} ]
		# , {'rot_90':bconfig.rot_90} ]
		random.shuffle(augment_list)
		current_aug = random.choice(augment_list)
		# for key in augment_list[0].keys():
		if 'horizontal_flips' in current_aug.keys():
			aug = get_aug([HorizontalFlip(p=1)])
			bbox_after_aug ,augmented = apply_augmentations(img_data_aug,img,aug,class_m)
		elif 'vertical_flips'  in current_aug.keys():
			aug = get_aug([VerticalFlip(p=1)])
			bbox_after_aug ,augmented = apply_augmentations(img_data_aug,img,aug,class_m)
		elif 'transpose'  in current_aug.keys():
			aug = get_aug([Transpose(p=1)])
			bbox_after_aug ,augmented = apply_augmentations(img_data_aug,img,aug,class_m)
			# elif 'rot_90' == key:
			# 	aug = get_aug([Rotate(limit=90,p=1)])
			# 	bbox_after_aug ,augmented = apply_augmentations(img_data_aug,img,aug,class_m)

	if len(img_data_aug_ori['bboxes']) == len(bbox_after_aug):
		for bbox,bbox_aug in zip(img_data_aug_ori['bboxes'],bbox_after_aug):
			bbox['x1'] = abs(bbox_aug['x1'])
			bbox['y1'] = abs(bbox_aug['y1'])
			bbox['x2'] = abs(bbox_aug['x2'])
			bbox['y2'] = abs(bbox_aug['y2'])
	else:
		img_data_aug_ori = img_data
		augmented = {'image':img}

	img_data_aug_ori['width'] = augmented['image'].shape[1]
	img_data_aug_ori['height'] = augmented['image'].shape[0]
	return img_data_aug_ori, augmented['image']

