import cv2
import pandas as pd
import os

from albumentations import (
    HorizontalFlip, BboxParams ,Transpose ,Rotate,Compose,VerticalFlip,normalize_bbox
)

def write_bbox(Bbox,image,img_name):
    x1 = Bbox['xmin']
    y1 = Bbox['ymin']
    x2 = Bbox['xmax']
    y2 = Bbox['ymax']
    rect_img = cv2.rectangle(image,(x1 ,y1),(x2, y2),(0,0,0),3)
    cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/final_combined/Augmentations_test/fungus_ori/{0}.png'.format(img_name),rect_img)

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 
                                               min_visibility=min_visibility,label_fields=['category_id']))

def current_augmentattions():
    image = cv2.imread('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/190304-73-599-he-20_5_22_2560_11264_3072_11776.png')
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

def visualize_bbox(img, bbox, class_id, class_idx_to_name,thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    color = (0, 0, 0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (0,0,0),3)
    TEXT_COLOR = (255, 255, 255)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img

def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
        cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/albumtations_after_h.png',img)

def albu_aug():
    image = cv2.imread('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/190304-73-599-he-20_5_22_2560_11264_3072_11776.png')
    # 165,5,323,124 - 165 , 5 , 158 , 119
    
    # aug = get_aug([HorizontalFlip(p=1)])
    # aug = get_aug([VerticalFlip(p=1)])
    # aug = get_aug([Rotate(limit=90,p=1)])
    annotations = {'image': image, 'bboxes': [[165,5, 158, 119]],'category_id': [1]} # This requires bboxes in x,y,w,h format
    category_id_to_name = {1: 'bacteria'}
    aug = get_aug([Transpose(p=1)])
    augmented = aug(**annotations)
    visualize(augmented, category_id_to_name)


    # augmented = aug(**annotations)
    # img = augmented['image'].copy()
    # for idx, bbox in enumerate(augmented['bboxes']):
    #     x_min, y_min, w, h = bbox
    #     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    #     # x_min, x_max, y_min, y_max = bbox
    #     bbox_list = [  {'xmin':165,'ymin':5,'xmax':323,'ymax':124} ]
    #     # write_bbox(bbox_list[0],image,'190304-73-599-he-20_5_22_2560_11264_3072_11776_before_h')
    #     # print("augmented",.shape)
    #     rect_img = cv2.rectangle(augmented['image'],(x_min ,y_min),(x_max, y_max),(0,0,0),3)
    #     cv2.imwrite('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/albumtations_after_v.png',rect_img)

if __name__ == "__main__":

    # current_augmentattions() # currently training frcnn onn this one
    # albu_aug() # albumentations exp

# beforehori = {'height': 512, 'imageset': 'trainval', 'filepath': '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_he/current_ones/1-152_2_13_2043_13285_3067_14309_1.png', 
# 'width': 512, 
# 'bboxes': [{'class': 'Bacteria', 'x2': 309, 'y1': 0, 'y2': 331, 'x1': 0},

# {'class': 'Bacteria', 'x2': 319, 'y1': 280, 'y2': 402, 'x1': 114}, 
# {'class': 'Bacteria', 'x2': 84, 'y1': 393, 'y2': 474, 'x1': 25}, 
# {'class': 'Bacteria', 'x2': 60, 'y1': 473, 'y2': 512, 'x1': 0}, 
# {'class': 'Bacteria', 'x2': 70, 'y1': 296, 'y2': 337, 'x1': 15}]}

# bbox_after_aug = [{'x2': 512.0, 'y1': 0.0, 'y2': 331.0, 'x1': 203.0}, 

# {'x2': 398.0, 'y1': 280.0, 'y2': 402.0, 'x1': 193.0}, 
# {'x2': 487.0, 'y1': 393.0, 'y2': 474.0, 'x1': 428.0}, 
# {'x2': 512.0, 'y1': 473.0, 'y2': 512.0, 'x1': 452.0}, 
# {'x2': 497.0, 'y1': 296.0, 'y2': 337.0, 'x1': 442.0}]

# img_data_aug {'filepath': '/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_he/current_ones/1-437_6_6_6131_6131_7155_7155_0.png',
#  'imageset': 'trainval', 'width': 512,
#   'bboxes': [{'y1': 188, 'class': 'Bacteria', 'x1': 387, 'y2': 324, 'x2': 490}, 
#   {'y1': 84, 'class': 'Bacteria', 'x1': 210, 'y2': 153, 'x2': 375}, 
#   {'y1': 278, 'class': 'Bacteria', 'x1': 0, 'y2': 369, 'x2': 103}, 
#   {'y1': 144, 'class': 'Bacteria', 'x1': 231, 'y2': 224, 'x2': 349},
#    {'y1': 11, 'class': 'Bacteria', 'x1': 377, 'y2': 68, 'x2': 429}], 'height': 512}

#     ## Horizontal Aug
#     image = cv2.imread('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/training_data_he/current_ones/1-152_2_13_2043_13285_3067_14309_1.png')
#     Bbox = {'xmax': 309, 'ymin': 0, 'ymax': 331, 'xmin': 0}
#     write_bbox(Bbox,image,'1-152_2_13_2043_13285_3067_14309_beofer')
#     annotations = {'image': image, 'bboxes': [[0,0, 309, 331]],'category_id': [1]} # This requires bboxes in x,y,w,h format
#     category_id_to_name = {1: 'bacteria'}
#     aug = get_aug([HorizontalFlip(p=1)])
#     augmented = aug(**annotations)
#     Bbox = {'xmax': 512, 'ymin': 0, 'ymax': 331, 'xmin': 203}
#     write_bbox(Bbox,augmented['image'],'1-152_2_13_2043_13285_3067_14309_after_h')

    df = pd.read_csv('/home/kalpit/User/palash_konverge/projects/ability/vgg_annotator/training_data/final_combined_txt/fungus_april21bbox_anno.csv')
    for index, row in df.iterrows():
        image = cv2.imread(row['filename'])
        img_name = os.path.basename(row['filename']).split('.')[0]
        print(img_name)
        write_bbox(row,image,img_name)

