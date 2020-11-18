from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import traceback
from keras import backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model

from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn import pascal_voc_parser

# CWCD: https://github.com/kentaroy47/frcnn-from-scratch-with-keras
sys.setrecursionlimit(40000)

parser = OptionParser()
dir1 = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/Faster_RCNN_2/good_weights/'
CheckpointLimit = 20


parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="simple")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of RoIs to process at once.", default=6)
parser.add_option("--network", dest="network",
                  help="Base network to use. Supports vgg or resnet50.", default='resnet50')
# parser.add_option("--hf", dest="horizontal_flips",
#                   help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=True)
# parser.add_option("--vf", dest="vertical_flips",
#                   help="Augment with vertical flips in training. (Default=false).", action="store_true", default=True)
# parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
#                   action="store_true", default=True)
parser.add_option("--num_epochs", type="int", dest="num_epochs",
                  help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config_march9.pickle")
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

print('options.parser',options.parser)
if options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError(
        "Command line option parser must be one of 'pascal_voc' or 'simple'")

# if not options.train_path:   # if filename is not given
#     parser.error(
#         'Error: path to training data must be specified. Pass --path to command line')
# if options.parser == 'pascal_voc':
#     from keras_frcnn.pascal_voc_parser import get_data
# elif options.parser == 'simple':
#     from keras_frcnn.simple_parser import get_data
# else:
#     raise ValueError(
#         "Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()
# C.use_horizontal_flips = bool(options.horizontal_flips)
# C.use_vertical_flips = bool(options.vertical_flips)
# C.rot_90 = bool(options.rot_90)
# print('C.use_horizontal_flips',C.use_horizontal_flips)
# print('C.use_horizontal_flips',C.use_vertical_flips)
# print('C.use_horizontal_flips',C.rot_90)

C.num_rois = int(options.num_rois)

if options.network == 'vgg':
    from keras_frcnn import vgg as nn
    C.network = 'vgg'
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
elif options.network == 'resnet101':
    from keras_frcnn import resnet as nn
    C.network = 'resnet101'
else:
    print('Not a valid model')
    raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    if C.network == 'resnet101':
        print('Starting training on resnet101')
        C.base_net_weights = '/home/kalpit/User/palash_konverge/projects/ability/frcnn_model/Faster_RCNN_2/resnet101_imagenet_1000_no_top.h5'
        # nn.get_weight_path(depth=101)
    else:
        C.base_net_weights = nn.get_weight_path()

print('options.train_path',options.train_path)
all_imgs, classes_count, class_mapping = get_data(options.train_path)

# print('all_imgs',all_imgs[0])
print('classes_count',classes_count)
print('class_mapping',class_mapping)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval' or s['imageset'] == 'test' ]

# val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
# print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(
    train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), class_mapping, mode='train')
# data_gen_val = data_generators.get_anchor_gt(
#     val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
# shared_layers = nn.resnet_graph(img_input, architecture='resnet101')[-1]
# shared_layers = nn.nn_base(img_input, trainable=True)


if options.network == 'vgg':
    shared_layers = nn.nn_base(img_input, trainable=True)
elif options.network == 'resnet50':
    shared_layers = nn.resnet_graph(img_input, architecture='resnet50')[-1]
elif options.network == 'resnet101':
    print('shared_layers resnet101')
    shared_layers = nn.resnet_graph(img_input, architecture='resnet101')[-1]
else:
    print('Not a valid model')
    raise ValueError

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(
    classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
    # print('model_classifier',model_classifier.summary())
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = SGD(lr=0.001, momentum=0.15, decay=1e-6)
optimizer_classifier = SGD(lr=0.001, momentum=0.15, decay=1e-6)
optimizer_all = SGD(lr=0.001, momentum=0.9, decay=1e-6)

model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(
    num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(
    len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer=optimizer_all, loss='mae')

# print(model_classifier.summary())
# print(model_rpn.summary())
# print(model_all.summary())
print('current trainig imgs',len(train_imgs))
epoch_length = len(train_imgs) - 1
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('class_mapping_inv',class_mapping_inv)
print('Starting training')

vis = True

EpochsSinceCheckpoint = 0

for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(
                    sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                # print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))

                if mean_overlapping_bboxes == 0:
                    pass
                    # print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            # print('startig X Y')
            X, Y, img_data = next(data_gen_train)
            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(
            ), use_regr=True, overlap_thresh=0.8, max_boxes=10)

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(
                R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(
                        pos_samples, C.num_rois//2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(
                        neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    try:
                        selected_neg_samples = np.random.choice(
                            neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                    except:
                        continue

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch(
                [X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                        ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(
                    sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print(
                        'Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()
                # model_all.save_weights(
                #     dir1+"saved-model-march11-" + str(epoch_num) + '_' + str(curr_loss) + ".hdf5")

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(
                            best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(
                        dir1+"fungus-res101-17-april-" + str(epoch_num) + '_' + str(curr_loss) + ".hdf5")
                    EpochsSinceCheckpoint = 0
                else:
                    EpochsSinceCheckpoint += 1

                print("Loss hasn't changed for {0} epochs. LR for model_all is {1}".format(EpochsSinceCheckpoint, K.get_value(model_all.optimizer.lr)))

                if EpochsSinceCheckpoint >= CheckpointLimit:
                    print("Losses did not change for {0} epochs. \
						Reducing LR for model_all from {1} to {2}".format(CheckpointLimit, K.get_value(model_all.optimizer.lr), K.get_value(model_all.optimizer.lr)*0.5))

                    K.set_value(model_all.optimizer.lr,
                                K.get_value(model_all.optimizer.lr)*0.5)
                    K.set_value(model_classifier.optimizer.lr,
                                K.get_value(model_all.optimizer.lr)*0.5)
                    K.set_value(model_rpn.optimizer.lr,
                                K.get_value(model_all.optimizer.lr)*0.5)
                    EpochsSinceCheckpoint = 0
                break

        except Exception as e:
            # print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')
