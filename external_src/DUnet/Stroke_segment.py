from keras.engine.training_arrays import predict_loop
import numpy as np
import os, sys
from model import *
from Statistics import *
from data_load import *
import matplotlib.pyplot as plt
sys.path.insert(0, '../stroke-lesion-segmentation/src')
from eval_utils import *
from log_utils import log
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
import argparse

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

def train_data_generator(h5_file_path, batch_size, augmentation):
    i = 0
    file = h5py.File(h5_file_path, 'r')
    imgs = file['data']
    labels = file['label']

    indexes = list(range(imgs.shape[0]))
    total_length = len(indexes)
    aug_gen = ImageDataGenerator(**augmentation)

    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(indexes)

            current_img = imgs[[indexes[i]],...]
            current_label = labels[[indexes[i]],...]
            aug_iter = aug_gen.flow(x=current_img, y=current_label, batch_size=1)
            current_img, current_label = aug_iter.__next__()
            batch_img.append(current_img[0,...])
            batch_label.append(current_label[0,...])
            i = (i + 1) % total_length

        yield np.array(batch_img), np.array(batch_label)

def create_train_data_generator(h5_file_path, batch_size, aug):
    return train_data_generator(h5_file_path, batch_size, aug)

def val_data_generator(h5_file_path, augmentation, batch_size=1):
    i = 0
    file = h5py.File(h5_file_path, 'r')
    imgs = file['data_val']
    labels = file['label_val']
    aug_gen = ImageDataGenerator(**augmentation)

    indexes = list(range(imgs.shape[0]))
    num_of_slices = len(indexes)

    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            current_img = imgs[[indexes[i]],...]
            current_label = labels[[indexes[i]],...]
            aug_iter = aug_gen.flow(x=current_img, y=current_label, batch_size=1)
            current_img, current_label = aug_iter.__next__()
            batch_img.append(current_img[0,...])
            batch_label.append(current_label[0,...])
            i = (i + 1) % num_of_slices
        yield np.array(batch_img), np.array(batch_label)


def create_val_date_generator(h5_file_path, aug, batch_size=1):
    return val_data_generator(h5_file_path, aug, batch_size)

def dice_coef_loss(y_true, y_pred):
    return -1 * dice_coef(y_true, y_pred)

if __name__ == "__main__":
    SMALL_LESION_PATH = "../stroke-lesion-segmentation/data/atlas_lesion_segmentation_small_lesions/validation_small_lesion_idxs.txt"
    val_data_list = []
    val_image_paths = "data_split/atlas/train_val/validation/val_scans.txt"
    with open(val_image_paths, "r") as val_data_file:
        val_data_list = val_data_file.readlines()

    val_data_list = ["data/atlas/atlas_standard/{}.nii.gz".format(data.rstrip()[:-4]) for data in val_data_list]
    val_label_list = [["{}_LesionSmooth_stx.nii.gz".format(data.rstrip()[:-15]),
                        "{}_LesionSmooth_1_stx.nii.gz".format(data.rstrip()[:-15])] for data in val_data_list]
    print(val_data_list)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    path_h5_save = './h5/'
    output_path = './trained_model/'
    dataset_name = '0.8'
    load_weight = 'trained_model/DUnet/trial1/DUnet-126-0.69929.hdf5'
    trial = 1
    # load_weight = ''
    mode = 'detect'  # use 'train' or 'detect'
    img_size = [192, 192]
    batch_size = 36
    lr = 1e-5
    gpu_used = 2
    if not os.path.exists("trained_model/DUnet/trial{}/outputs".format(trial)):
        os.mkdir("trained_model/DUnet/trial{}/outputs".format(trial))
    log_path = "trained_model/DUnet/trial{}/val_result.txt".format(trial)
    visual_paths = ['trained_model/DUnet/trial{}/flipped_visual/sl'.format(trial)]
    small_lesion_only = True

    model = D_Unet()
    h5_name = 'DUnet'
    output_path += h5_name+'/'+'trial3/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.summary()
    model = multi_gpu_model(model, gpus=gpu_used)
    model.compile(optimizer=SGD(lr=lr), loss=EML, metrics=[dice_coef])

    if load_weight != '':
        print('loading：', load_weight)
        model.load_weights(load_weight, by_name=True)
    else:
        print('no loading weight!')

    if mode == 'train':
        train_h5 = 'h5/train'
        file_train = h5py.File(train_h5, 'r')
        original = file_train['data']
        label = file_train['label']
        # label = h5['label_change']
        val_h5 = 'h5/validation'
        file_val = h5py.File(val_h5, 'r')
        original_val = file_val['data_val']
        label_val = file_val['label_val']
        # label_val = h5['label_val_change']


        num_train_steps = math.floor(len(original) / batch_size)
        num_val_steps = math.floor(len(original_val) / batch_size)

        print('training data:' + str(len(original)) + '  validation data:' + str(len(original_val)))

        # print('using:', str(time.time() - time_start) + 's\n')
        time_start = time.time()
        data_gen_args = dict(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, rotation_range=20,
                             horizontal_flip=True, featurewise_center=True, featurewise_std_normalization=True)
        data_gen_args_validation = dict(featurewise_center=True, featurewise_std_normalization=True)

        #data_gen_args = dict()
        #data_gen_args_validation = dict()

        # train_datagen = ImageDataGenerator(**data_gen_args)
        # train_datagen_label = ImageDataGenerator(**data_gen_args)
        # validation_datagen = ImageDataGenerator(**data_gen_args_validation)
        # validation_datagen_label = ImageDataGenerator(**data_gen_args_validation)

        # image_generator = train_datagen.flow(original, batch_size=batch_size, seed=1)
        # mask_generator = train_datagen_label.flow(label, batch_size=batch_size, seed=1)
        # image_generator_val = validation_datagen.flow(original_val, batch_size=batch_size, seed=1)
        # mask_generator_val = validation_datagen_label.flow(label_val, batch_size=batch_size, seed=1)

        # train_generator = zip(image_generator, mask_generator)
        # validation_generator = zip(image_generator_val, mask_generator_val)

        checkpointer = ModelCheckpoint(output_path + h5_name + '-{epoch:02d}-{val_dice_coef:.5f}.hdf5', verbose=1, save_best_only=True, period=1)
        History=model.fit_generator(create_train_data_generator(train_h5, batch_size, aug=data_gen_args),
                                    epochs=150,
                                    steps_per_epoch=num_train_steps,
                                    callbacks=[checkpointer],
                                    validation_data=create_val_date_generator(val_h5, aug=data_gen_args_validation, batch_size=2),
                                    validation_steps=num_val_steps,
                                    verbose=1)

    elif mode == 'detect':
        print('loading testing-data...')
        # h5 = h5py.File('./h5/validation', 'r')
        # original = h5['data_val']
        # label = h5['label_val']

        # label_val_change = h5['label_val_change']
        print('load data done!')

        model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[TP, TN, FP, FN, dice_coef, recall_ori, precision_ori])

        dice_list = []
        recall_list = []
        precision_list = []

        tp = 0
        fp = 0
        fn = 0

        #'''
        small_lesion_idx = parse_small_lesion_txt(SMALL_LESION_PATH)

        total_time = 0.0
        dice_scores = np.zeros(27)
        ious = np.zeros(27)
        precisions = np.zeros(27)
        recalls = np.zeros(27)
        patient_all = []

        sl_dice_scores = np.zeros(27)
        sl_ious = np.zeros(27)
        sl_precisions = np.zeros(27)
        sl_recalls = np.zeros(27)
        sl_dice_from_ext = np.zeros(27)

        # Store per-class metrics (lesion, non-lesion, mean)
        per_class_dices = np.zeros((27, 2))
        per_class_ious = np.zeros((27, 2))
        per_class_precisions = np.zeros((27, 2))
        per_class_recalls = np.zeros((27, 2))

        sl_per_class_dices = np.zeros((27, 2))
        sl_per_class_ious = np.zeros((27, 2))
        sl_per_class_precisions = np.zeros((27, 2))
        sl_per_class_recalls = np.zeros((27, 2))

        data_gen_args_validation = dict(featurewise_center=True, featurewise_std_normalization=True)
        datagen = ImageDataGenerator(**data_gen_args_validation)


        for idx, dir in enumerate(val_data_list):
            print(dir)
            patient_all.append(dir[32:45])
            data = nib.load(val_data_list[idx])
            data = data.get_fdata()
            data = np.array(data)
            data = data.transpose((2, 1, 0))
            label = np.zeros_like(data)
            p1, p2 = val_label_list[idx]
            if os.path.exists(p1):
                img1 = nib.load(p1)
                img1 = np.array(img1.get_fdata())
                img1 = img1.transpose((2, 1, 0))
                label = label + img1
            if os.path.exists(p2):
                img2 = nib.load(p2)
                img2 = np.array(img2.get_fdata())
                img2 = img2.transpose((2, 1, 0))
                label = label + img2
            data = np.array(data, dtype=float)
            label = np.array(label, dtype=bool)

            data = np.uint8(np.multiply(data, 2.55))
            label = np.uint8(label)
            data_resize = []
            label_resize = []
            for i in range(len(data)):
                data_resize_single = Image.fromarray(data[i]).crop((10, 40, 190, 220))
                data_resize_single = data_resize_single.resize((192,192), Image.ANTIALIAS)
                data_resize_single = np.asarray(data_resize_single)

                label_resize_single = Image.fromarray(label[i]).crop((10, 40, 190, 220))
                label_resize_single = label_resize_single.resize((192,192), Image.ANTIALIAS)
                label_resize_single = np.asarray(label_resize_single)

                label_resize.append(label_resize_single)
                data_resize.append(data_resize_single)

            data = np.array(data_resize, dtype=float)
            label = np.array(label_resize, dtype=int)
            data = data - data.min()
            data = data / data.max()
            assert np.isfinite(label).all()
            print(label.max())
            label = label - label.min()
            label = label / label.max()

            label = np.expand_dims(label, axis=-1)
            data = data_toxn(data, 4)

            predict_patient = []
            print(data.shape)

            for i in range(data.shape[0]):
                curr_data = data[[i], ...]
                dg = datagen.flow(curr_data, batch_size=1)
                data[i, ...] = dg.__next__()

            t = time.time()
            for i in range(0, len(data), 3):

                predict = model.predict(data[[i, i+1, i+2],...])
                predict_patient.append(predict)
            total_time += time.time() - t
            # predict2 = model.predict(data, batch_size=3)
            predict_patient = np.concatenate(predict_patient, axis=0)
            print(predict_patient.shape)
            print(predict_patient.max())
            # assert np.all(np.equal(predict, predict2))
            output_sig = predict_patient.copy()
            predict_patient[predict_patient >= 0.5] = 1
            predict_patient[predict_patient < 0.5] = 0

            small_lesion_patient = small_lesion_idx[dir[32:45]]
            sl_out = predict_patient[list(small_lesion_patient),...]
            sl_gt = label[list(small_lesion_patient),...]

            dice_scores[idx] = dice_score(predict_patient, label)
            ious[idx] = IOU(predict_patient, label)
            precisions[idx] = precision(predict_patient, label)
            recalls[idx] = recall(predict_patient, label)

            class_histogram = compute_prediction_hist(np.int64(label).flatten(), np.int64(predict_patient).flatten(), 2)
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


            save_path = "trained_model/DUnet/trial{}/outputs/{}.npy".format(trial, os.path.split(dir)[1][:-15])
            print("saving to {}".format(save_path))
            np.save(save_path, predict_patient)

            if visual_paths is not None:
                if not os.path.exists(visual_paths[0]):
                    os.makedirs(visual_paths[0])
                with open("../stroke-lesion-segmentation/validation/atlas/trainval/atlas_trainval_val_scans.txt",'r') as val_idx_file:
                    val_list = val_idx_file.readlines()
                val_list = [p.rstrip()[37:50] for p in val_list]
                val_dict = {p:i for i, p in enumerate(val_list)}
                patient = dir[32:45]
                site = patient[:5]

                for i in range(predict_patient.shape[0]):
                    if small_lesion_only and i not in small_lesion_patient:
                        continue
                    chunk = data[[i],...]
                    chunk = chunk[...,2]
                    chunk = np.expand_dims(chunk, axis=1)
                    output_sigmoid = output_sig[i]
                    output_sigmoid = np.squeeze(output_sigmoid, axis=2)

                    assert chunk.shape[0] == 1, ('Batch size should be 1', chunk.shape)
                    ground_truth_2d = label[i]
                    ground_truth_2d = ground_truth_2d.astype(np.int16)
                    ground_truth_2d = np.squeeze(ground_truth_2d, axis=2)

                    ground_truth_2d = ground_truth_2d.transpose((1,0))
                    chunk = chunk.transpose((0,1,3,2))
                    output_sigmoid = output_sigmoid.transpose((1,0))
                    output_seg = np.squeeze(predict_patient[i], 2).transpose((1,0))

                    save_prediction_img(
                        chunk=chunk,
                        idx=val_dict[patient],
                        chunk_idx=i,
                        output_segmentation=output_seg,
                        output_segmentation_soft=output_sigmoid,
                        ground_truth_2d=ground_truth_2d,
                        visual_paths=visual_paths
                    )
            # for data_point in val_data_list:
            # calculate mean value of each metric across all scans
        print("total time taken: {}".format(total_time))
        print("per patient time: {}".format(total_time / 27))

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(ious)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        sl_mean_dice = np.mean(sl_dice_scores)
        sl_mean_iou = np.mean(sl_ious)
        sl_mean_precision = np.mean(sl_precisions)
        sl_mean_recall = np.mean(sl_recalls)

        # tuples of n_classes + 1 elements; means of lesion_class, non_lesion_class, all
        mean_per_class_dice = perclass2mean(per_class_dices)
        mean_per_class_iou = perclass2mean(per_class_ious)
        mean_per_class_precision = perclass2mean(per_class_precisions)
        mean_per_class_recall = perclass2mean(per_class_recalls)

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


