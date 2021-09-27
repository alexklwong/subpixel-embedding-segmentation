# Code for KiU-Net
# Author: Jeya Maria Jose
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os, sys
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from arch.ae import kiunet,kinetwithsk,unet,autoencoder
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import time
from matplotlib import cm
import matplotlib.image as mpimg
sys.path.insert(0, 'src/')
import utils
import global_constants as settings
from eval_utils import *
from log_utils import log

def parse_small_lesion_txt(path, n_samples=None):
    '''
    Given a path to a .txt file saved in create_small_lesion_maps(),
        return list of list of idxs of small indices

    Arg(s):
        path: str
            List to .txt file saved in the format above
        n_samples : int or None
            Optional argument used as a sanity check to make sure we have the correct number
            of idx lists
    Returns:
        list[list[int]] : each outer list corresponds with a patient scan.
        The order corresponds with the order in the .txt file
            Elements of inner list correspond with chunk idx of ONLY small lesions
    '''

    lines = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            idx, mri_id, n_small_lesions = line[0:3]

            small_lesion_idxs = line[3:]
            if len(small_lesion_idxs) == 0:
                continue
            try:
                small_lesion_idxs = [int(idx) for idx in small_lesion_idxs]
            except ValueError:
                continue
            set_small_lesion_idxs = set(small_lesion_idxs)
            assert len(set_small_lesion_idxs) == len(small_lesion_idxs)

            lines[mri_id] = set_small_lesion_idxs

    if n_samples is not None:
        assert n_samples == len(lines)
    return lines

def min_max_normalization(scan, dataset_min, dataset_max):
    '''
    Normalize scan by adding the minimum value to entire dataset,
      calculate new maximum, and divide each scan by new max

    Arg(s):
        scan: numpy
            tensor to normalize
        dataset_min: float
            minimum value of dataset
        dataset_max: float
            maximum value of dataset
    Returns:
        numpy: normalized scan with same shape as scan
    '''
    return ((scan - dataset_min) / (dataset_max - dataset_min))

def save_prediction_img(chunk,
                        idx,
                        chunk_idx,
                        output_segmentation,
                        output_segmentation_soft,
                        ground_truth_2d,
                        visual_paths,
                        lesion_sizes=None):
    '''
    For each 2D prediction, save original scan (BW), prediction (color, BW), and GT (color, BW)
    Arg(s):
        chunk : numpy array of D x H x W
            input scan
        idx : int
            patient scan index (0 indexed)
        chunk_idx : int
            which chunk of patient scan (0 indexed)
        output_segmentation : numpy array of H x W
            2d segmentation prediction
        output_segmentation_soft : numpy array
            softmax prediction
        ground_truth : numpy array H x W
            ground truth annotation for chunk of scan
        visual_paths : list[str]
            1 element for each modality for path to save each modality img
        lesion_sizes : list[float] (optional)
            list of relative lesion sizes
    Returns:
        None
    '''

    nb_2d_scans = chunk.shape[1]

    lesion_size = np.sum(ground_truth_2d)

    if lesion_size > 0 and lesion_sizes is not None:
        lesion_sizes.append(np.round(lesion_size / ground_truth_2d.size, 4))

    for modality_id in range(len(visual_paths)):
        # extract center 2d scan:
        if len(chunk.shape) == 5:
            chunk_mod = chunk[0, nb_2d_scans // 2, modality_id]
        elif len(chunk.shape) == 4:
            chunk_mod = chunk[0, nb_2d_scans // 2]

        # color lesions:
        viridis = plt.get_cmap('viridis', 256)
        output_segmentation_colored = viridis(output_segmentation_soft)
        output_segmentation_gray = cm.gray(output_segmentation.astype(np.float32))

        viridis = plt.get_cmap('viridis', 2)
        ground_truth_2d_colored = viridis(ground_truth_2d)
        ground_truth_2d_gray = cm.gray(ground_truth_2d.astype(np.float32))

        # overlay on gray scans:
        overlay_eps = 0.5
        # chunk_mod = normalize(chunk_mod)
        chunk_mod = min_max_normalization(
            chunk_mod,
            dataset_min=np.min(chunk_mod),
            dataset_max=np.max(chunk_mod)
        )
        chunk_mod_pred = cm.gray(chunk_mod)
        chunk_mod_gt = cm.gray(chunk_mod)

        chunk_mod_pred[output_segmentation_soft > overlay_eps, :] = output_segmentation_colored[output_segmentation_soft > overlay_eps, :]
        chunk_mod_gt[ground_truth_2d > overlay_eps, :] = ground_truth_2d_colored[ground_truth_2d > overlay_eps, :]

        # store predictions as png
        image_scan_gray_path = os.path.join(
            visual_paths[modality_id],
            'scan_gray',
            'scange_gray_patient%d' % (idx),
            'scan%d.png' % (chunk_idx))
        image_pred_color_path = os.path.join(
            visual_paths[modality_id],
            'pred_color_overlay',
            'pred_color_patient%d' % (idx),
            'scan%d.png' % (chunk_idx))
        image_pred_gray_path = os.path.join(
            visual_paths[modality_id],
            'pred_gray',
            'pred_gray_patient%d_' % (idx),
            'scan%d.png' % (chunk_idx))
        image_gt_color_path = os.path.join(
            visual_paths[modality_id],
            'gt_color_overlay',
            'gt_color_patient%d' % (idx),
            'scan%d.png' % (chunk_idx))
        image_gt_gray_path = os.path.join(
            visual_paths[modality_id],
            'gt_gray',
            'gt_gray_patient%d' % (idx),
            'scan%d.png' % (chunk_idx))

        visual_outputs = [
            (image_scan_gray_path, cm.gray(chunk_mod), 'gray'),
            (image_pred_color_path, chunk_mod_pred, 'viridis'),
            (image_pred_gray_path, output_segmentation_gray, 'gray'),
            (image_gt_color_path, chunk_mod_gt, 'viridis'),
            (image_gt_gray_path, ground_truth_2d_gray, 'gray')
        ]

        for (visual_path, visual, colormap) in visual_outputs:

            if not os.path.exists(os.path.dirname(visual_path)):
                os.makedirs(os.path.dirname(visual_path))

            mpimg.imsave(visual_path, visual, cmap=colormap, vmin=0.0, vmax=1.0)


parser = argparse.ArgumentParser(description='KiU-Net')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_dataset',  type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 5)

parser.add_argument('--modelname', default='off', type=str,
                    help='model name')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: on)')

parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='overcomplete_udenet', type=str,
                    help='model name')
parser.add_argument('--direc', default='./special', type=str,
                    help='directory to save the outputs')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str, help='path to the model to be restored')
parser.add_argument('--visual_paths', type=str, nargs='+', help='path to store the visualization results')
parser.add_argument('--small_lesion_only', action='store_true', default=False, help='whether to store visualization for small lesion images only')
parser.add_argument('--validation_path', type=str, default='validation/atlas/trainval/atlas_trainval_val_annotations.txt', 
    help='paths to validation data')
parser.add_argument('--small_lesion_idx', type=str, default='data/atlas_lesion_segmentation_small_lesions/validation_small_lesion_idxs.txt',
    help='path to small lesion idx')
parser.add_argument('--n_samples', type=int, default=27, help='number of patients in the validation set')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes gets predicted (default: 2)')
parser.add_argument('--log_path', type=str, help='the path to log the statistics')
parser.add_argument('--image_size', nargs='+', type=int, default=(224, 192), help='the size of the image to reshape to')

args = parser.parse_args()

direc = args.direc
modelname = args.modelname
loaddirec = args.loaddirec

def add_noise(img):
    noise = torch.randn(img.size()) * 0.1
    noisy_img = img + noise.cuda()

    return noisy_img
     

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
# train_dataset = ImageToImage2D(args.train_dataset, tf_val)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
# dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "unet":
    model = unet()
elif modelname =="autoencoder":
    model =autoencoder()
elif modelname == "kiunet":
    model = kiunet()
elif modelname == "kinetwithsk":
    model = kinetwithsk()
elif modelname == "kinet":
    model = kinet()
elif modelname == "pspnet":
    model = psp.PSPNet(layers=5, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=False).cuda()

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)
# model.apply(weight_init)
# print(model)
bestdice=0

criterion = LogNLLLoss()

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)

metric_list = MetricList({'jaccard': partial(jaccard_index),
                          'f1': partial(f1_score)})

with open(args.validation_path, 'r') as val_data_order:
    val_list = val_data_order.readlines()
val_list = [p.rstrip()[37:50] for p in val_list]
val_dict = {p:i for i, p in enumerate(val_list)}
if args.visual_paths is not None:
    if not os.path.exists(args.visual_paths[0]):
        os.makedirs(args.visual_paths[0])

small_lesion_idx = parse_small_lesion_txt(args.small_lesion_idx)

model.load_state_dict(torch.load(loaddirec))
model.eval()

outputs = {}
time_taken = 0
with torch.no_grad():
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
        # print(batch_idx)
        if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
        else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        patient_name = image_filename[:13]
        patient_idx = int(image_filename[14:17])
        small_lesion_patient = small_lesion_idx[patient_name]
        print(patient_name, patient_idx)
        if patient_name not in outputs:
            outputs[patient_name] = [None] * 189

        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        t = time.time()
        y_out = model(X_batch)
        y_out = torch.softmax(y_out, dim=1)
        time_taken += time.time() - t

        tmp2 = y_batch.detach().cpu().numpy()
        tmp = y_out.detach().cpu().numpy()
        output_soft = tmp.copy()
        tmp[tmp>=0.5] = 1
        tmp[tmp<0.5] = 0
        tmp2[tmp2>0] = 1
        tmp2[tmp2<=0] = 0
        tmp2 = tmp2.astype(int)
        tmp = tmp.astype(int)
        outputs[patient_name][patient_idx] = tmp

        visualize = (not args.small_lesion_only) or (args.small_lesion_only and (patient_idx in small_lesion_patient))
        if len(args.visual_paths) > 0 and visualize:
            chunk = X_batch.cpu().numpy()
            output_sigmoid = output_soft[0, 1]

            assert chunk.shape[0] == 1, ('Batch size should be 1', chunk.shape)
            ground_truth_2d = tmp2

            ground_truth_2d = ground_truth_2d[0,...].transpose((1,0))
            chunk = chunk.transpose((0,1,3,2))
            output_sigmoid = output_sigmoid.transpose((1,0))
            output_seg = tmp[0,1].transpose((1,0))

            save_prediction_img(
                chunk=chunk,
                idx=val_dict[patient_name],
                chunk_idx=patient_idx,
                output_segmentation=output_seg,
                output_segmentation_soft=output_sigmoid,
                ground_truth_2d=ground_truth_2d,
                visual_paths=args.visual_paths
            )

        # print(np.unique(tmp2))
        yHaT = tmp
        yval = tmp2

        epsilon = 1e-20
        
        del X_batch, y_batch,tmp,tmp2, y_out

        # count = count + 1
        yHaT[yHaT==1] =255
        yval[yval==1] =255
        fulldir = direc+"/outputs/"

print("total time taken for prediction: {}".format(time_taken))
print("per sample prediction time: {}".format((time_taken) / 27))

if not os.path.exists(direc):
    os.mkdir(direc)

for name, output in outputs.items():
    assert not any(elem is None for elem in output)
    print(name)
    save_file = os.path.join(direc, "{}.npy".format(name))
    np.save(save_file, np.array(output))

# added code below to run the evaluation on the output. 
# Create an 1D array for each metric that holds the value for each metric
n_samples = args.n_samples
n_classes = args.n_classes
VALIDATION_PATH = args.validation_path
PREDICTION_PATH = args.direc
log_path = args.log_path
size = args.image_size

# Create an 1D array for each metric that holds the value for each metric
dice_scores = np.zeros(n_samples)
ious = np.zeros(n_samples)
precisions = np.zeros(n_samples)
recalls = np.zeros(n_samples)
dice_from_ext = np.zeros(n_samples)
outputs_all = []
patient_all = []
gt_all = []

# Store per-class metrics (lesion, non-lesion, mean)
per_class_dices = np.zeros((n_samples, n_classes))
per_class_ious = np.zeros((n_samples, n_classes))
per_class_precisions = np.zeros((n_samples, n_classes))
per_class_recalls = np.zeros((n_samples, n_classes))

gt_paths = []
with open(VALIDATION_PATH, 'r') as val_anno:
    gt_paths = val_anno.readlines()
gt_paths = [gt_path.rstrip() for gt_path in gt_paths]

output_paths = os.listdir(PREDICTION_PATH)
for idx, path in enumerate(output_paths):
    
    patient = path[:-4]
    patient_all.append(patient)
    output_load_path = os.path.join(PREDICTION_PATH, path)
    print(output_load_path)

    output_np = np.load(output_load_path)
    print(output_np.shape)
    output_np = np.squeeze(output_np, axis=1)
    output_np = output_np[:, 1, :, :]
    output_np[output_np >= 0.5] = 1
    output_np[output_np < 0.5] = 0

    gt_np = None
    found = False
    for gt_path in gt_paths:
        if patient in gt_path:
            gt_np = np.load("{}".format(gt_path))
            found = True
            break
    assert found

    gt_np = gt_np.transpose((2,1,0))
    gt_np[gt_np > 0] = 1
    gt_np[gt_np <= 0] = 0
    gt_resized = np.zeros((189, 128, 128))
    for i,gt_one in enumerate(gt_np):
        gt_one = gt_one * 255
        gt_one = gt_one.astype(np.uint8)
        gt_one = Image.fromarray(gt_one)
        gt_one = gt_one.resize((128, 128), resample=Image.NEAREST)
        gt_one = np.array(gt_one)
        gt_one[gt_one > 0] = 1
        gt_one[gt_one <= 0] = 0
        gt_resized[i, :, :] = gt_one

    print(gt_resized.shape)
    print(output_np.shape)

    dice_scores[idx] = dice_score(output_np, gt_resized)
    ious[idx] = IOU(output_np, gt_resized)
    precisions[idx] = precision(output_np, gt_resized)
    recalls[idx] = recall(output_np, gt_resized)

    class_histogram = compute_prediction_hist(np.int64(gt_resized).flatten(), np.int64(output_np).flatten(), n_classes)
    per_class_dices[idx] = per_class_dice(class_histogram)
    per_class_ious[idx] = per_class_iou(class_histogram)
    per_class_recalls[idx] = per_class_recall(class_histogram)
    per_class_precisions[idx] = per_class_precision(class_histogram)


# calculate mean value of each metric across all scans
mean_dice = np.mean(dice_scores)
mean_iou = np.mean(ious)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

# tuples of n_classes + 1 elements; means of lesion_class, non_lesion_class, all
mean_per_class_dice = perclass2mean(per_class_dices)
mean_per_class_iou = perclass2mean(per_class_ious)
mean_per_class_precision = perclass2mean(per_class_precisions)
mean_per_class_recall = perclass2mean(per_class_recalls)

best_results = {}
best_results['step'] = 0
best_results['dice'] = mean_dice
best_results['iou'] = mean_iou
best_results['precision'] = mean_precision
best_results['recall'] = mean_recall

best_results['per_class_dice'] = mean_per_class_dice
best_results['per_class_iou'] = mean_per_class_iou
best_results['per_class_precision'] = mean_per_class_precision
best_results['per_class_recall'] = mean_per_class_recall

# Store current results as latest
best_results['step_last'] = 0
best_results['dice_last'] = mean_dice
best_results['iou_last'] = mean_iou
best_results['precision_last'] = mean_precision
best_results['recall_last'] = mean_recall

best_results['per_class_dice_last'] = mean_per_class_dice
best_results['per_class_iou_last'] = mean_per_class_iou
best_results['per_class_precision_last'] = mean_per_class_precision
best_results['per_class_recall_last'] = mean_per_class_recall

# log current results
log('Validation results: ', log_path)

log('Current step:', log_path)
log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
    'Step', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
    0, mean_dice, mean_iou, mean_precision, mean_recall), log_path)

log('{:>10}'.format('Per class metrics:'), log_path)
log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Class', 'dice', 'iou', 'precision', 'recall'), log_path)
log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
    'Non-Lesion', mean_per_class_dice[0], mean_per_class_iou[0], mean_per_class_precision[0], mean_per_class_recall[0]), log_path)
log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
    'Lesion', mean_per_class_dice[1], mean_per_class_iou[1], mean_per_class_precision[1], mean_per_class_recall[1]), log_path)
log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
    'Mean', mean_per_class_dice[2], mean_per_class_iou[2], mean_per_class_precision[2], mean_per_class_recall[2]), log_path)

log("", log_path)
log("per patient stats:", log_path)
for i in range(len(dice_scores)):
    patient = patient_all[i]
    log('{:>14}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Patient', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
    log('{:>14}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        patient, dice_scores[i], ious[i], precisions[i], recalls[i]), log_path)
