import argparse
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
import utils
import global_constants as settings
from eval_utils import *
from log_utils import log
import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=27, help='number of patients in the validation set')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes gets predicted (default: 2)')
parser.add_argument('--log_path', type=str, help='the path to log the statistics')
parser.add_argument('--image_size', nargs='+', type=int, default=(224, 192), help='the size of the image to reshape to')
parser.add_argument('--validation_path', type=str, default='validation/atlas/trainval/atlas_trainval_val_annotations.txt',
    help='paths to validation data')
parser.add_argument('--direc', default='./special', type=str,
                help='directory to save the outputs')
args = parser.parse_args()

PREDICTION_PATH = args.direc
VALIDATION_PATH = args.validation_path
n_samples = args.n_samples
n_classes = args.n_classes
size = args.image_size
log_path = args.log_path

if __name__ == "__main__":
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
                gt_np = np.load("../stroke-lesion-segmentation/{}".format(gt_path))
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

        # outputs_all.append(output_np)
        # gt_all.append(gt_np)

        # # The below code are for D-Unet
        # gt_np = gt_np.transpose((2,1,0))
        # gt_np = np.uint8(gt_np)
        # gt_resized = []
        # for i in range(gt_np.shape[0]):
        #     label_resize_single = Image.fromarray(gt_np[i, ...]).crop((10, 40, 190, 220))
        #     label_resize_single = label_resize_single.resize(size, Image.ANTIALIAS)
        #     label_resize_single = np.asarray(label_resize_single)

        #     gt_resized.append(label_resize_single)
        # gt_resized = np.array(gt_resized)
        # gt_resized = gt_resized - gt_resized.min()
        # gt_resized = gt_resized / gt_resized.max()

        # dice_ext, _, _, _, _, _, _ = utils.get_score_for_one_patient(gt_resized, output_np)
        # dice_from_ext[idx] = dice_ext

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

    # print("dice from get_score_for_one_patient: {}".format(np.mean(dice_from_ext)))

    # print("statistics from get_score_from_all_slices: ")
    # gt_all = np.concatenate(gt_all, axis=0)
    # outputs_all = np.concatenate(outputs_all, axis=0)
    # score_record = utils.get_score_from_all_slices(gt_all, outputs_all)
    # for key in score_record.keys():
    #     print('In fold ', 0, ', average', key, ' value is: \t ', np.mean(score_record[key]))


    log("", log_path)
    log("per patient stats:", log_path)
    for i in range(len(dice_scores)):
        patient = patient_all[i]
        log('{:>14}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Patient', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
        log('{:>14}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            patient, dice_scores[i], ious[i], precisions[i], recalls[i]), log_path)
