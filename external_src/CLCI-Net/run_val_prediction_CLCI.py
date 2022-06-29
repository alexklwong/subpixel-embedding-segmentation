import argparse
import sys
import os
sys.path.insert(0, '../stroke-lesion-segmentation/src')
import utils
import global_constants as settings
from eval_utils import *
from log_utils import log
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_path', type=str, default='', help='the paths to the directory that contains the output prediction')
parser.add_argument('--validation_path', type=str, default='testing/atlas/traintest/atlas_test_ground_truths.txt', help='the path to the validation data')
parser.add_argument('--log_path', type=str, help='the path to log the statistics')
parser.add_argument('--small_lesion_only', action='store_true', help='whether to store visualization for small lesion samples only')
parser.add_argument('--visual_path', type=str, help='the path to store visualization')

args = parser.parse_args()

PREDICTION_PATH = args.prediction_path
VALIDATION_PATH = args.validation_path
n_samples = 27
n_classes = 2
log_path = args.log_path
SMALL_LESION_PATH = "testing/atlas/traintest/atlas_test_small_lesion_map_indices.txt"
small_lesion_only = args.small_lesion_only
visual_paths = [args.visual_path]

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

if __name__ == "__main__":
    if not os.path.exists(visual_paths[0]):
        os.makedirs(visual_paths[0])
    with open(VALIDATION_PATH,'r') as val_idx_file:
        val_list = val_idx_file.readlines()
    val_list = [p.rstrip()[36:49] for p in val_list]
    val_dict = {p:i for i, p in enumerate(val_list)}

    small_lesion_idx = parse_small_lesion_txt(SMALL_LESION_PATH)
    print(small_lesion_idx)

    # Create an 1D array for each metric that holds the value for each metric
    patient_all = []
    dice_scores = np.zeros(n_samples)
    ious = np.zeros(n_samples)
    precisions = np.zeros(n_samples)
    recalls = np.zeros(n_samples)
    dice_from_ext = np.zeros(n_samples)
    outputs_all = []
    gt_all = []

    sl_dice_scores = np.zeros(27)
    sl_ious = np.zeros(27)
    sl_precisions = np.zeros(27)
    sl_recalls = np.zeros(27)
    sl_dice_from_ext = np.zeros(27)

    # Store per-class metrics (lesion, non-lesion, mean)
    per_class_dices = np.zeros((n_samples, n_classes))
    per_class_ious = np.zeros((n_samples, n_classes))
    per_class_precisions = np.zeros((n_samples, n_classes))
    per_class_recalls = np.zeros((n_samples, n_classes))

    sl_per_class_dices = np.zeros((27, 2))
    sl_per_class_ious = np.zeros((27, 2))
    sl_per_class_precisions = np.zeros((27, 2))
    sl_per_class_recalls = np.zeros((27, 2))

    gt_paths = []
    with open(VALIDATION_PATH, 'r') as val_anno:
        gt_paths = val_anno.readlines()
    gt_paths = ["{}".format(gt_path.rstrip()) for gt_path in gt_paths]

    output_paths = os.listdir(PREDICTION_PATH)
    for idx, path in enumerate(output_paths):

        patient = path[:-4]
        patient_all.append(patient)
        small_lesion_patient = small_lesion_idx[patient]
        output_load_path = os.path.join(PREDICTION_PATH, path)
        print(output_load_path)

        output_np = np.load(output_load_path)
        print(output_np.shape)

        # This one for CLCI-net
        output_np = np.squeeze(output_np, axis=-1)
        output_np = np.squeeze(output_np, axis=1)
        output_sig = output_np.copy()
        output_np[output_np >= 0.5] = 1
        output_np[output_np < 0.5] = 0

        gt_np = None
        found = False
        for gt_path in gt_paths:
            if patient in gt_path:
                gt_np = np.load(gt_path)
                found = True
                break
        assert found

        gt_np = gt_np.transpose((2,1,0))
        gt_np = np.uint(gt_np)
        gt_resized = gt_np[: ,5:229, 11:187]
        gt_resized[gt_resized > 0] = 1
        gt_resized[gt_resized <= 0] = 0

        print(gt_resized.shape)
        print(output_np.shape)

        outputs_all.append(output_np)
        gt_all.append(gt_resized)

        sl_gt = gt_resized[list(small_lesion_patient),...]
        sl_out = output_np[list(small_lesion_patient),...]

        dice_scores[idx] = dice_score(output_np, gt_resized)
        ious[idx] = IOU(output_np, gt_resized)
        precisions[idx] = precision(output_np, gt_resized)
        recalls[idx] = recall(output_np, gt_resized)

        class_histogram = compute_prediction_hist(np.int64(gt_resized).flatten(), np.int64(output_np).flatten(), n_classes)
        per_class_dices[idx] = per_class_dice(class_histogram)
        per_class_ious[idx] = per_class_iou(class_histogram)
        per_class_recalls[idx] = per_class_recall(class_histogram)
        per_class_precisions[idx] = per_class_precision(class_histogram)

        sl_dice_scores[idx] = dice_score(sl_out, sl_gt)
        sl_ious[idx] = IOU(sl_out, sl_gt)
        sl_precisions[idx] = precision(sl_out, sl_gt)
        sl_recalls[idx] = recall(sl_out, sl_gt)

        sl_class_histogram = compute_prediction_hist(np.int64(sl_gt).flatten(), np.int64(sl_out).flatten(), 2)
        sl_per_class_dices[idx] = per_class_dice(sl_class_histogram)
        sl_per_class_ious[idx] = per_class_iou(sl_class_histogram)
        sl_per_class_recalls[idx] = per_class_recall(sl_class_histogram)
        sl_per_class_precisions[idx] = per_class_precision(sl_class_histogram)

        if len(visual_paths) > 0:
            site = patient[:5]
            input_path = "data/atlas_spin/{}/{}/{}_t1w_stx.npy".format(site, patient, patient)
            image = np.load(input_path)
            image = image.transpose((2,1,0))
            image = image[: ,5:229, 11:187]
            print(image.shape)

            for i in range(output_np.shape[0]):
                if small_lesion_only and i not in small_lesion_patient:
                    continue
                chunk = image[[i],...]
                chunk = np.expand_dims(chunk, axis=1)
                output_sigmoid = output_sig[i]

                assert chunk.shape[0] == 1, ('Batch size should be 1', chunk.shape)
                ground_truth_2d = gt_resized[i]
                ground_truth_2d = ground_truth_2d.astype(np.int16)

                ground_truth_2d = ground_truth_2d.transpose((1,0))
                chunk = chunk.transpose((0,1,3,2))
                output_sigmoid = output_sigmoid.transpose((1,0))
                output_seg = output_np[i].transpose((1,0))

                save_prediction_img(
                    chunk=chunk,
                    idx=val_dict[patient],
                    chunk_idx=i,
                    output_segmentation=output_seg,
                    output_segmentation_soft=output_sigmoid,
                    ground_truth_2d=ground_truth_2d,
                    visual_paths=visual_paths
                )

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

    sl_mean_dice = np.mean(sl_dice_scores)
    sl_mean_iou = np.mean(sl_ious)
    sl_mean_precision = np.mean(sl_precisions)
    sl_mean_recall = np.mean(sl_recalls)

    sl_mean_per_class_dice = perclass2mean(sl_per_class_dices)
    sl_mean_per_class_iou = perclass2mean(sl_per_class_ious)
    sl_mean_per_class_precision = perclass2mean(sl_per_class_precisions)
    sl_mean_per_class_recall = perclass2mean(sl_per_class_recalls)

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

    log('', log_path)
    log('validation results for small lesion samples: ', log_path)

    log('Current step:', log_path)
    log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
        'Step', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        0, sl_mean_dice, sl_mean_iou, sl_mean_precision, sl_mean_recall), log_path)

    log('{:>10}'.format('Per class metrics:'), log_path)
    log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Class', 'dice', 'iou', 'precision', 'recall'), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        'Non-Lesion', sl_mean_per_class_dice[0], sl_mean_per_class_iou[0], sl_mean_per_class_precision[0], sl_mean_per_class_recall[0]), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        'Lesion', sl_mean_per_class_dice[1], sl_mean_per_class_iou[1], sl_mean_per_class_precision[1], sl_mean_per_class_recall[1]), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        'Mean', sl_mean_per_class_dice[2], sl_mean_per_class_iou[2], sl_mean_per_class_precision[2], sl_mean_per_class_recall[2]), log_path)


    log("", log_path)
    log("per patient stats:", log_path)
    for i in range(len(dice_scores)):
        patient = patient_all[i]
        log('{:>14}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Patient', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
        log('{:>14}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            patient, dice_scores[i], ious[i], precisions[i], recalls[i]), log_path)

