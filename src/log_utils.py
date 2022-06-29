import os, torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
from data_utils import min_max_normalization


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console

    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, "w+") as o:
                o.write(s+'\n')
        else:
            with open(filepath, "a+") as o:
                o.write(s+'\n')

def colorize(T, colormap='magma'):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps

    Arg(s):
        T : tensor
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)

    # Convert to numpy array and transpose
    T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap and transpose back
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)
    color = np.transpose(color, (0, 3, 1, 2))

    # Convert back to tensor
    return torch.from_numpy(color.astype(np.float32))

def save_MRI_prediction_img(chunk,
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
        chunk : numpy[float32]
            1 x C x D x H x W input scan, D refers to number of modalities
            OR
            1 x C x H x W input scan (RGB: C = 3)
        idx : int
            patient scan index (0 indexed)
        chunk_idx : int
            which chunk of patient scan (0 indexed)
        output_segmentation : numpy[float32]
            H x W segmentation prediction
        output_segmentation_soft : numpy[float32]
            softmax prediction
        ground_truth : numpy[float32]
            H x W ground truth annotation for chunk of scan
        visual_paths : list[str]
            1 element for each modality for path to save each modality img
        lesion_sizes : list[float] (optional)
            list of relative lesion sizes
    '''

    n_channel = chunk.shape[1]

    lesion_size = np.sum(ground_truth_2d)

    if lesion_size > 0 and lesion_sizes is not None:
        lesion_sizes.append(np.round(lesion_size / ground_truth_2d.size, 4))

    for modality_id in range(len(visual_paths)):
        # extract center 2d scan:
        if len(chunk.shape) == 5:
            chunk_mod = chunk[0, n_channel  // 2, modality_id]
        elif len(chunk.shape) == 4:
            chunk_mod = chunk[0, n_channel  // 2]

        # color lesions:
        viridis = plt.get_cmap('viridis', 256)
        output_segmentation_colored = viridis(output_segmentation_soft)
        output_segmentation_gray = cm.gray(output_segmentation.astype(np.float32))

        viridis = plt.get_cmap('viridis', 2)
        ground_truth_2d_colored = viridis(ground_truth_2d)
        ground_truth_2d_gray = cm.gray(ground_truth_2d.astype(np.float32))

        # overlay on gray scans:
        overlay_eps = 0.5

        chunk_mod = min_max_normalization(
            chunk_mod,
            dataset_min=np.min(chunk_mod),
            dataset_max=np.max(chunk_mod))

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

def save_RGB_prediction_img(chunk,
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
        chunk : numpy[float32]
            1 x 3 x H x W input scan
            OR
            3 x H x W input scan
        idx : int
            patient scan index (0 indexed)
        chunk_idx : int
            which chunk of patient scan (0 indexed)
        output_segmentation : numpy[float32]
            H x W segmentation prediction
        output_segmentation_soft : numpy[float32]
            softmax prediction
        ground_truth : numpy[float32]
            H x W ground truth annotation for chunk of scan
        visual_paths : list[str]
            1 element for each modality for path to save each modality img
        lesion_sizes : list[float] (optional)
            list of relative lesion sizes
    '''
    lesion_size = np.sum(ground_truth_2d)

    if lesion_size > 0 and lesion_sizes is not None:
        lesion_sizes.append(np.round(lesion_size / ground_truth_2d.size, 4))

    for modality_id in range(len(visual_paths)):
        if len(chunk.shape) == 4:
            chunk = chunk[0, ...]

        if len(chunk.shape) != 3:
            raise ValueError("RGB visualizations only available for tensors with 3 or 4 dimensions. Received {}-dimension tensor".format(len(chunk.shape)))

        chunk = np.transpose(chunk, (1, 2, 0)).copy(order='C')
        # Add 1's in C dimension to convert from RGB -> RGBA
        alphas = np.ones((chunk.shape[0], chunk.shape[1], 1))
        chunk = np.concatenate([chunk, alphas], axis=-1)

        # color lesions:
        viridis = plt.get_cmap('viridis', 256)

        output_segmentation_colored = viridis(output_segmentation_soft)
        output_segmentation_gray = cm.gray(output_segmentation.astype(np.float32))
        viridis = plt.get_cmap('viridis', 2)
        ground_truth_2d_colored = viridis(ground_truth_2d)
        ground_truth_2d_gray = cm.gray(ground_truth_2d.astype(np.float32))

        # overlay on gray scans:
        overlay_eps = 0.5

        chunk = min_max_normalization(
            chunk,
            dataset_min=np.min(chunk),
            dataset_max=np.max(chunk))
        chunk_pred = np.copy(chunk, order='C')
        chunk_gt = np.copy(chunk, order='C')

        chunk_pred[output_segmentation_soft > overlay_eps, :] = output_segmentation_colored[output_segmentation_soft > overlay_eps, :]
        chunk_gt[ground_truth_2d > overlay_eps, :] = ground_truth_2d_colored[ground_truth_2d > overlay_eps, :]

        # concatenate the image, prediction, and ground truth overlays in one image
        combined = np.concatenate([chunk, chunk_pred, chunk_gt], axis=1)
        # Create paths to store images as .png
        image_scan_color_path = os.path.join(
            visual_paths[modality_id],
            'scan_color',
            'scan_color_patient%d.png' % (idx))
        image_pred_color_path = os.path.join(
            visual_paths[modality_id],
            'pred_color_overlay',
            'pred_color_patient%d.png' % (idx))
        image_pred_gray_path = os.path.join(
            visual_paths[modality_id],
            'pred_gray',
            'pred_gray_patient%d.png' % (idx))
        image_gt_color_path = os.path.join(
            visual_paths[modality_id],
            'gt_color_overlay',
            'gt_color_patient%d.png' % (idx))
        image_gt_gray_path = os.path.join(
            visual_paths[modality_id],
            'gt_gray',
            'gt_gray_patient%d.png' % (idx))
        combined_color_path = os.path.join(
            visual_paths[modality_id],
            'combined_color',
            'combined_color_patient%d.png' % (idx))

        visual_outputs = [
            (image_scan_color_path, chunk),
            (image_pred_color_path, chunk_pred),
            (image_pred_gray_path, output_segmentation_gray),
            (image_gt_color_path, chunk_gt),
            (image_gt_gray_path, ground_truth_2d_gray),
            (combined_color_path, combined)
        ]
        for (visual_path, visual) in visual_outputs:

            if not os.path.exists(os.path.dirname(visual_path)):
                os.makedirs(os.path.dirname(visual_path))
            mpimg.imsave(visual_path, visual, vmin=0.0, vmax=1.0)