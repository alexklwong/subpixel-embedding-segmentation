import cv2, time
import torch
import numpy as np
from sklearn import metrics
from data_utils import save_numpy_to_nii, save_numpy, get_n_chunk
from log_utils import log
import global_constants as settings


def dice_score(src, tgt):
    '''
    Calculate Dice Score

    Arg(s):
        src: numpy array of predicted segmentation
        tgt: numpy array of ground truth segmentation
    Returns:
        float : dice score ( 2 * (A intersect B)/(|A| + |B|))
    '''

    intersection = np.logical_and(src, tgt)
    total = src.sum() + tgt.sum()
    if total == 0:  # avoid divide by 0
        return 0.0
    return 2 * intersection.sum() / total

def IOU(src, tgt):
    '''
    Calculate Intersection Over Union (IOU)

    Arg(s):
        src: numpy
            numpy array of predicted segmentation
        tgt: numpy
            numpy array of ground truth segmentation
    Returns:
        float : intersection over union ((A intersect B)/ (A U B))
    '''

    intersection = np.logical_and(src, tgt)
    union = np.logical_or(src, tgt)
    if union.sum() == 0:  # avoid divide by 0
        return 0.0
    return intersection.sum() / union.sum()

def per_class_iou(hist):
    '''
    Calculate IOU per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        iou for the ith class.
    '''

    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # number of times a class is ground-truth or predicted:
    union = hist.sum(1) + hist.sum(0) - nb_of_tp

    # compute iou:
    iou_per_class = nb_of_tp / union

    return iou_per_class

def per_class_dice(hist):
    '''
    Calculate Dice per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        dice for the ith class.
    '''
    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # sum of times a class is ground-truth or predicted:
    denom = hist.sum(1) + hist.sum(0)

    # compute dice:
    dice_per_class = 2*nb_of_tp / denom

    return dice_per_class

def perclass2mean(per_class_stat):
    '''
    Calculate means per class (non-lesion, lesion, overall)

    Arg(s):
        per_class_stat : numpy
            2D array (N x 2) where columns represent statistic for non-lesion and lesion classes
    Returns:
        tuple[float] : mean values by column (non_lesion_mean, lesion_mean, overall_mean)
    '''
    non_lesion_mean = np.nanmean(per_class_stat[:, 0])
    lesion_mean = np.nanmean(per_class_stat[:, 1])
    overall_mean = np.nanmean(per_class_stat)

    return non_lesion_mean, lesion_mean, overall_mean

def compute_prediction_hist(label, pred, num_classes):
    '''
    Given labels, predictions, and the number of classes, compute a histogram summary

    Arg(s):
        label : numpy
            1D numpy array of ground truth labels
        pred : numpy
            1D numpy array of predicted labels with the same size as label
        num_classes : int
            number of classes
    Returns:
        numpy : 2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class.
    '''
    # Sanity checks
    assert len(label) == len(pred), (len(label), len(pred))
    assert label.ndim == 1, label.ndim
    assert pred.ndim == 1, pred.ndim

    '''
    mask is a boolean vector of length len(label) to ignore the invalid pixels
    e.g. when there is an ignored class which is assigned to 255.
    '''
    mask = (label >= 0) & (label < num_classes)

    '''
    label_pred_1d is a 1D vector for valid pixels, where gt labels are modulated
    with num_classes to store them in the rows of the prediction histogram.
    Goal is to encode number of times each (label, pred) pair is seen.
    '''
    label_pred_1d = num_classes * label[mask].astype(int) + pred[mask]

    # convert set of (label, pred) pairs into a 1D histogram of size num_classes**2:
    hist_1d = np.bincount(label_pred_1d, minlength=num_classes**2)

    # convert 1d histogram to 2d histogram:
    hist_2d = hist_1d.reshape(num_classes, num_classes)
    assert hist_2d.shape[0] == num_classes, (hist_2d.shape[0], num_classes)
    assert hist_2d.shape[1] == num_classes, (hist_2d.shape[1], num_classes)

    return hist_2d

def precision(src, tgt):
    '''
    Calculate precision

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : precision = (true positives / (true positives + false positives))
    '''
    src_positives = np.sum(src)   # = true positives + false positives
    true_positives = np.sum(np.logical_and(src, tgt))
    if src_positives == 0:
        return 0.0

    return true_positives / src_positives

def per_class_precision(hist):
    '''
    Calculate precision per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        precision for the ith class.
    '''
    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # number of times a class is predicted:
    nb_pred = hist.sum(0)

    # compute precision:
    precision_per_class = nb_of_tp / nb_pred

    return precision_per_class

def recall(src, tgt):
    '''
    Calculate recall

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : recall = (true positives / (true positives + false negatives))
    '''
    true_positives = np.sum(np.logical_and(src, tgt))
    inverse_src = np.logical_not(src)

    # false negative is prediction labeled as not lesion but actually is lesion
    false_negatives = np.sum(np.logical_and(inverse_src, tgt))
    if (true_positives + false_negatives) == 0:
        return 0.0

    return true_positives / (true_positives + false_negatives)

def per_class_recall(hist):
    '''
    Calculate recall per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        recall for the ith class.
    '''
    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # number of times a class is the ground truth:
    nb_gt = hist.sum(1)

    # compute recall:
    recall_per_class = nb_of_tp / nb_gt

    return recall_per_class

def accuracy(src, tgt):
    '''
    Calculate accuracy

    Arg(s):
        src: numpy
            numpy array of predicted segmentation
        tgt: numpy
            numpy array of ground truth segmentation
    Returns:
        float : accuracy (|src - tgt| / tgt)
    '''
    true_positives = np.sum(np.logical_and(src, tgt))
    inverse_src = np.logical_not(src)
    inverse_tgt = np.logical_not(tgt)
    true_negatives = np.sum(np.logical_and(inverse_src, inverse_tgt))

    total = np.sum(np.ones_like(tgt))
    return (true_positives + true_negatives) / total

def specificity(src, tgt):
    '''
    Calculate specificity

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : specificty = (true negatives / (true negatives + false positives))
    '''
    inverse_src = np.logical_not(src)
    inverse_tgt = np.logical_not(tgt)

    true_negatives = np.sum(np.logical_and(inverse_src, inverse_tgt))
    false_positives = np.sum(np.logical_and(src, inverse_tgt))

    # Check for divide by 0
    if true_negatives + false_positives == 0:
        return 0.0

    return true_negatives / (true_negatives + false_positives)

def f1(src, tgt):
    '''
    Calculate f1

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : f1 = 2 * precision * recall / (precision + recall)
    '''
    precision_metric = precision(src, tgt)
    recall_metric = recall(src, tgt)

    if precision_metric + recall_metric == 0:
        return 0.0

    return (2 * precision_metric * recall_metric) / (precision_metric + recall_metric)

def auc_roc(src, tgt):
    '''
    Calculate area under the curve of ROC curve

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : AUC of ROC
    '''
    # sklearn's function takes in flattened 1D arrays
    assert src.shape == tgt.shape
    if len(src.shape) != 1:
        src = src.flatten()
        tgt = tgt.flatten()
    # target cannot be all same element
    if len(np.unique(tgt)) == 1:
        return None
    return metrics.roc_auc_score(tgt, src)

def log_best_results(log_path, best_results):
    '''
    Logs the best results to log_path

    Arg(s):
        best_results : dictionary or None
            if None, return
            Otherwise assert best_results has the correct entries
        log_path : str
            Path to which to log to
    Returns:
        None
    '''

    entries = ['step', 'dice', 'iou', 'precision', 'recall']
    per_class_entries = ['per_class_dice', 'per_class_iou', 'per_class_precision', 'per_class_recall']
    hasRGBMetrics = 'accuracy' in best_results
    if hasRGBMetrics:
        entries += ['accuracy', 'specificity', 'f1', 'auc_roc']

    if best_results is None:
        return

    # Check correct formatting
    assert(entry in best_results for entry in entries)
    for entry in per_class_entries:
        assert entry in best_results
        assert len(best_results[entry]) == 3

    log('\nBest step:', log_path)
    log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
        'Best', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        best_results['step'], best_results['dice'], best_results['iou'], best_results['precision'], best_results['recall']), log_path)
    log('Per class metrics:', log_path)
    log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Class', 'dice', 'iou', 'precision', 'recall'), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        'Non-Lesion', best_results['per_class_dice'][0], best_results['per_class_iou'][0], best_results['per_class_precision'][0], best_results['per_class_recall'][0]), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        'Lesion', best_results['per_class_dice'][1], best_results['per_class_iou'][1], best_results['per_class_precision'][1], best_results['per_class_recall'][1]), log_path)
    log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
        'Mean', best_results['per_class_dice'][2], best_results['per_class_iou'][2], best_results['per_class_precision'][2], best_results['per_class_recall'][2]), log_path)
    if hasRGBMetrics:
        log('RGB Metrics:', log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Step', 'Accuracy', 'Spec.', 'F1', 'AUC_ROC'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            best_results['step'], best_results['accuracy'], best_results['specificity'], best_results['f1'], best_results['auc_roc']), log_path)
    log('---***---', log_path)

def average_flip(model, output, image, flip_type):
    '''
    Perform predictions and average the predictions for the original and flipped images.

    Arg(s):
        model : torch model
            torch network to feed images.
        output : torch tensor
            prediction of the model for the original image, (... H X W).
        image : torch tensor
            the original image, (... H X W).
        flip_type : str
            flip type for augmentation (combination of 'horizontal_test', 'vertical_test', and 'both_test')
    Returns:
        numpy : the average of the predictions, (... X H X W).
    '''

    is_hflip = 'horizontal_test' in flip_type
    is_vflip = 'vertical_test' in flip_type
    is_both_flip = 'both_test' in flip_type

    nb_pred = 1
    image = image.cpu().numpy()
    output = output.cpu().numpy()

    if is_hflip:
        # horizontally flip the image:
        image_hflip = image[..., ::-1].copy()
        # convert the image into torch tensor:
        image_hflip = torch.from_numpy(image_hflip).cuda()
        # get the prediction for the flipped image:
        output_hflip = model.forward(image_hflip.cuda())[-1].cpu().data.numpy()
        # flip the prediction back:
        output_hflip = output_hflip[..., ::-1].copy()
        # sum the predictions:
        output += output_hflip
        nb_pred += 1

    if is_vflip:
        # vertically flip the image:
        image_vflip = image[..., ::-1, :].copy()
        # convert the image into torch tensor:
        image_vflip = torch.from_numpy(image_vflip).cuda()
        # get the prediction for the flipped image:
        output_vflip = model.forward(image_vflip.cuda())[-1].cpu().data.numpy()
        # flip the prediction back:
        output_vflip = output_vflip[..., ::-1, :].copy()
        # sum the predictions:
        output += output_vflip
        nb_pred += 1

    if is_both_flip:
        # vertically and horizontally flip the image:
        image_bothflip = image[..., ::-1, :].copy()
        image_bothflip = image_bothflip[..., ::-1].copy()
        # convert the image into torch tensor:
        image_bothflip = torch.from_numpy(image_bothflip).cuda()
        # get the prediction for the flipped image:
        output_bothflip = model.forward(image_bothflip.cuda())[-1].cpu().data.numpy()
        # flip the prediction back:
        output_bothflip = output_bothflip[..., ::-1, :].copy()
        output_bothflip = output_bothflip[..., ::-1].copy()
        # sum the predictions:
        output += output_bothflip
        nb_pred += 1

    # compute the average of the predictions:
    output /= nb_pred

    # convert the output into torch tensor:
    output = torch.from_numpy(output).cuda()

    return output

def validateMRI(model,
             dataloader,
             transforms,
             step,
             log_path,
             save_prediction_img,
             ground_truths=None,
             test_time_flip_type='',
             n_chunk=1,
             n_classes=2,
             dataset_means=[0, 0, 0, 0, 0],
             best_results=None,
             summary_writer=None,
             output_paths=[],
             visual_paths=[]):
    '''
    Given validation data and ground truths and a model, create predictions
        and return (optionally best) results

    Arg(s):
        model : SPiNModel
            Trained model to make prediction
        dataloader : torch.utils.data.DataLoader
            Validation dataloader for scans (NOT shuffled)
        transforms : Transforms
            transformations for the dataloader
        step : int
            Which step of training (mostly for best_results updates and tensorboard)
        log_path : string
            Path to logging file
        save_prediction_img : function
            MRI or RGB version to save image predictions
        ground_truths : list[numpy[int64]]
            ground truth annotations corresonding to same order as data_loader
        test_time_flip_type : str
            flipping types used at test time
        n_chunk : int
            number of chunks we grab to feed into model per prediction
        n_classes : int
            Number of classes we are predicting (default = 2 in this case)
        dataset_means  : list[int]
            list of mean values for each data modality
        best_results : dict[str, float32]
            Optional parameter to compare these results to
        summary_writer : tensorboard SummaryWriter or None
            Optional parameter for validation SummaryWriter
        output_paths : list[str]
            Output paths, if empty, no nii file is saved
        visual_paths : list[str]
            if not empty, visualize and store predictions as png
    Returns:
        best_results : dict[str, float32]
            If best_results was not None, return the better of this validate output
            and best_results. Otherwise return these results.
    '''

    n_samples = len(dataloader)

    # Choose a sample to log to tensorboard
    summary_input_scans = []
    summary_output_logits = []
    summary_ground_truths = []

    # Store prediction times for each patient
    prediction_times = np.zeros(n_samples)

    if ground_truths is not None:
        ground_truth_shape = ground_truths[0].shape

        height = ground_truth_shape[0]
        width = ground_truth_shape[1]

    if len(output_paths) > 0:
        assert len(output_paths) == n_samples, (len(output_paths), n_samples)

    # Create an 1D array for each metric that holds the value for each metric
    dice_scores = np.zeros(n_samples)
    ious = np.zeros(n_samples)
    precisions = np.zeros(n_samples)
    recalls = np.zeros(n_samples)

    # Store per-class metrics (lesion, non-lesion, mean)
    per_class_dices = np.zeros((n_samples, n_classes))
    per_class_ious = np.zeros((n_samples, n_classes))
    per_class_precisions = np.zeros((n_samples, n_classes))
    per_class_recalls = np.zeros((n_samples, n_classes))

    # Iterate through each scan
    for idx, scan in enumerate(dataloader):

        if ground_truths is not None:
            ground_truth = ground_truths[idx]
            patient_prediction = np.zeros_like(ground_truth)
        else:
            channel, height, width = scan.shape[1], scan.shape[-2], scan.shape[-1]
            patient_prediction = np.zeros((height, width, channel))

        patient_prediction_time = 0

        # Iterate through each chunk of each scan, calculate and store metric
        for chunk_idx in range(scan.shape[1]):

            # Scan shape: 1 x C x D x H x W
            chunk = get_n_chunk(
                scan,
                chunk_idx,
                n_chunk,
                constants=dataset_means,
                is_torch=True,
                input_type='BCDHW')

            # Move chunk to CUDA to same device as model
            chunk = chunk.to(model.device)

            [chunk] = transforms.transform(
                images_arr=[chunk],
                random_transform_probability=0.0)

            start_time = time.time()

            # Forward through super resolution and segmentation model
            output_logits = model.forward(chunk)

            # average predictions for horizontally/vertically flipped images:
            output_logits = average_flip(model, output_logits[-1], chunk, test_time_flip_type)

            chunk_prediction_time = time.time() - start_time

            # Save scan, segmentation, and ground truth chunk for tensorboard
            if chunk_idx == 90 and summary_writer is not None and ground_truths is not None:
                if np.random.rand() < 0.10 or idx == n_samples - 1:
                    ground_truth_chunk = ground_truth[..., chunk_idx]
                    ground_truth_chunk = cv2.resize(
                        ground_truth_chunk,
                        chunk.shape[-2:][::-1],
                        interpolation=cv2.INTER_NEAREST)
                    ground_truth_chunk = torch.from_numpy(ground_truth_chunk)

                    # Move ground truth to same device as model
                    ground_truth_chunk = ground_truth_chunk.to(model.device)
                    summary_input_scans.append(torch.squeeze(torch.squeeze(chunk, axis=0), axis=1))
                    summary_output_logits.append(torch.squeeze(torch.squeeze(output_logits, axis=0), axis=0))
                    summary_ground_truths.append(ground_truth_chunk)

            # Take probability maps and turn into 1 segmentation map, convert to ints
            output_sigmoid = torch.sigmoid(output_logits)
            output_segmentation = torch.where(
                output_sigmoid < 0.5,
                torch.zeros_like(output_sigmoid),
                torch.ones_like(output_sigmoid)).long()

            # Move the prediction to CPU to convert to numpy
            output_segmentation = output_segmentation.cpu().numpy()

            output_segmentation = np.squeeze(output_segmentation)

            # Resize prediction to ground truth annotation size
            output_segmentation = cv2.resize(
                output_segmentation,
                (width, height),
                interpolation=cv2.INTER_NEAREST)

            assert patient_prediction[:, :, chunk_idx].shape == output_segmentation.shape

            # Append the segment_prediction to the patient's entire prediction
            patient_prediction[:, :, chunk_idx] = output_segmentation

            if len(visual_paths) > 0:
                output_sigmoid = output_sigmoid[0, 0].cpu().data.numpy()

                # Resize soft prediction to ground truth annotation size
                output_sigmoid = cv2.resize(
                    output_sigmoid,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST)

                assert chunk.shape[0] == 1, ('Batch size should be 1', chunk.shape)
                ground_truth_2d = ground_truth[:, :, chunk_idx]

                save_prediction_img(
                    chunk=chunk,
                    idx=idx,
                    chunk_idx=chunk_idx,
                    output_segmentation=output_segmentation,
                    output_segmentation_soft=output_sigmoid,
                    ground_truth_2d=ground_truth_2d,
                    visual_paths=visual_paths)

            # Accumulate amount of time for predicting this patient's scan
            patient_prediction_time += chunk_prediction_time

        # Store patient's prediction time in array
        prediction_times[idx] = patient_prediction_time

        if len(output_paths) > 0:
            # Data type should be unsigned short
            patient_prediction = patient_prediction.astype(np.ushort)

            # save predictions in nii
            save_numpy_to_nii(patient_prediction, output_paths[idx])

        if ground_truths is not None:
            # Compute and store metrics for each patient
            dice_scores[idx] = dice_score(patient_prediction, ground_truth)
            ious[idx] = IOU(patient_prediction, ground_truth)
            precisions[idx] = precision(patient_prediction, ground_truth)
            recalls[idx] = recall(patient_prediction, ground_truth)

            if len(visual_paths) > 0:

                log('{:>9}  {:>9}  {:>9}  {:>9}  {:>9}'.format(
                    'Patient', 'Dice', 'IOU', 'Precision', 'Recall'),
                    log_path)
                log('{:>9}  {:9.3f}  {:9.3f}  {:9.3f}  {:9.3f}'.format(
                    idx, dice_scores[idx], ious[idx], precisions[idx], recalls[idx]),
                    log_path)

            class_histogram = compute_prediction_hist(ground_truth.flatten(), patient_prediction.flatten(), n_classes)
            per_class_dices[idx] = per_class_dice(class_histogram)
            per_class_ious[idx] = per_class_iou(class_histogram)
            per_class_recalls[idx] = per_class_recall(class_histogram)
            per_class_precisions[idx] = per_class_precision(class_histogram)

    # Calculate average patient prediction time
    mean_patient_prediction_time = np.mean(prediction_times)

    if ground_truths is not None:
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
        # Log metrics to tensorboard
        if summary_writer is not None:
            summary_input_scans = torch.stack(summary_input_scans, dim=0)
            summary_output_logits = torch.stack(summary_output_logits, dim=0)
            summary_ground_truths = torch.stack(summary_ground_truths, dim=0)

            model.log_summary(
                input_scan=summary_input_scans,
                output_logits=summary_output_logits,
                ground_truth=summary_ground_truths,
                scalar_dictionary={
                    'eval_metric_dice': mean_dice,
                    'eval_metric_iou': mean_iou,
                    'eval_metric_precision': mean_precision,
                    'eval_metric_recall': mean_recall
                },
                summary_writer=summary_writer,
                step=step,
                n_display=4)

        # Compare to best results
        means = {
            'dice': mean_dice,
            'iou': mean_iou,
            'precision': mean_precision,
            'recall': mean_recall,
            'per_class_dice': mean_per_class_dice,
            'per_class_iou': mean_per_class_iou,
            'per_class_precision': mean_per_class_precision,
            'per_class_recall': mean_per_class_recall,
        }
        best_results = update_best_results(best_results, step, means)

        # log current results
        log('Validation results: ', log_path)

        log('Mean patient prediction time: {:.3f} seconds'.format(mean_patient_prediction_time), log_path)

        log('Current step:', log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Step', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            step, mean_dice, mean_iou, mean_precision, mean_recall), log_path)

        log('{:>10}'.format('Per class metrics:'), log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Class', 'dice', 'iou', 'precision', 'recall'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Non-Lesion', mean_per_class_dice[0], mean_per_class_iou[0], mean_per_class_precision[0], mean_per_class_recall[0]), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Lesion', mean_per_class_dice[1], mean_per_class_iou[1], mean_per_class_precision[1], mean_per_class_recall[1]), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Mean', mean_per_class_dice[2], mean_per_class_iou[2], mean_per_class_precision[2], mean_per_class_recall[2]), log_path)

        # Log best results
        log_best_results(log_path, best_results)

    return best_results

def validateRGB(model,
                dataloader,
                transforms,
                step,
                log_path,
                save_prediction_img,
                ground_truths=None,
                test_time_flip_type='',
                n_chunk=1,
                n_classes=2,
                dataset_means=[0, 0, 0, 0, 0],
                best_results=None,
                summary_writer=None,
                output_paths=[],
                gt_shape=(256, 256),
                visual_paths=[],
                device=settings.DEVICE):
    '''
    Given validation data and ground truths and a model, create predictions
        and return (optionally best) results
    Arg(s):
        model : SPiNModel
            Trained model to make prediction
        dataloader : torch.utils.data.DataLoader
            Validation dataloader for scans (NOT shuffled)
        transforms : Transforms
            transformations for the data loader
        step : int
            Which step of training (mostly for best_results updates and tensorboard)
        log_path : string
            Path to logging file
        save_prediction_img : function
            MRI or RGB version to save image predictions
        ground_truths : list of numpy arrays
            ground truth annotations corresonding to same order as data_loader
        test_time_flip_type : string
            flipping types used at test time
        n_chunk : int
            number of chunks we grab to feed into model per prediction
        n_classes : int
            Number of classes we are predicting (default = 2 in this case)
        dataset_means  : list[int]
            list of mean values for each data modality
        best_results : dictionary or None
            Optional parameter to compare these results to
        summary_writer : tensorboard SummaryWriter or None
            Optional parameter for validation SummaryWriter
        output_paths : list of str
            Output paths, if empty, no nii file is saved.
        gt_shape : int tuple
            Ground truth size: (n_height, n_width).
        visual_paths : list of str
            if not empty, visualize and store predictions as png.

        device : string
            Which GPU device to move torch tensors to
    Returns:
        best_results : dictionary
            If best_results was not None, return the better of this validate output
            and best_results. Otherwise return these results.
    '''
    n_samples = len(dataloader)

    # Choose a sample to log to tensorboard
    summary_input_scans = []
    summary_output_logits = []
    summary_ground_truths = []

    if ground_truths is not None:
        summary_height = ground_truths[0].shape[0]
        summary_width = ground_truths[0].shape[1]
    else:  # in case no ground truth given. Need to set because RGB may have scans of different shapes
        summary_height = gt_shape[0]
        summary_height = gt_shape[1]

    # Store prediction times for each patient
    prediction_times = np.zeros(n_samples)

    if len(output_paths) > 0:
        assert len(output_paths) == n_samples, (len(output_paths), n_samples)

    # Create an 1D array for each metric that holds the value for each metric
    dice_scores = np.zeros(n_samples)
    ious = np.zeros(n_samples)
    precisions = np.zeros(n_samples)
    recalls = np.zeros(n_samples)
    accuracies = np.zeros(n_samples)
    specificities = np.zeros(n_samples)
    f1s = np.zeros(n_samples)
    auc_rocs = np.zeros(n_samples)

    # Store per-class metrics (lesion, non-lesion, mean)
    per_class_dices = np.zeros((n_samples, n_classes))
    per_class_ious = np.zeros((n_samples, n_classes))
    per_class_precisions = np.zeros((n_samples, n_classes))
    per_class_recalls = np.zeros((n_samples, n_classes))

    # Iterate through each scan
    for idx, scan in enumerate(dataloader):

        if ground_truths is not None:
            ground_truth = ground_truths[idx]
            height = ground_truth.shape[0]
            width = ground_truth.shape[1]
        else:
            height = scan.shape[0]
            width = scan.shape[1]

        lesion_sizes = []
        patient_prediction_time = 0
        chunk = scan

        # Move chunk to CUDA to same device as model
        chunk = chunk.to(model.device)

        [chunk] = transforms.transform(
            images_arr=[chunk],
            random_transform_probability=0.0)

        start_time = time.time()

        # Forward through super resolution and segmentation model
        output_logits = model.forward(chunk)

        # average predictions for horizontally/vertically flipped images:
        output_logits = average_flip(model, output_logits[-1], chunk, test_time_flip_type)

        chunk_prediction_time = time.time() - start_time

        # Save scan, segmentation, and ground truth chunk for tensorboard
        if summary_writer is not None and (np.random.rand() < 0.10 or idx == n_samples - 1):
            # Resize ground truth (cv2 requires numpy)
            ground_truth_tensorboard = cv2.resize(
                ground_truth,
                (summary_width, summary_height),
                interpolation=cv2.INTER_NEAREST)
            # Resize chunk (cv2 requires numpy)
            chunk_tensorboard = torch.squeeze(chunk, axis=0).cpu().numpy()  # 3 x H x W
            chunk_tensorboard = np.transpose(chunk_tensorboard, (1, 2, 0))  # H x W x 3 for cv2.resize()
            # Resize image
            chunk_tensorboard = cv2.resize(
                chunk_tensorboard,
                (summary_width, summary_height),
                interpolation=cv2.INTER_NEAREST)
            # Transpose back to 3 x H x W for tensorboard logging
            chunk_tensorboard = np.transpose(chunk_tensorboard, (2, 0, 1))
            # Resize output logits (cv2 requires numpy)
            output_logits_tensorboard = torch.squeeze(torch.squeeze(output_logits, axis=0), axis=0)
            output_logits_tensorboard = cv2.resize(
                output_logits_tensorboard.cpu().numpy(),
                (summary_width, summary_height),
                interpolation=cv2.INTER_NEAREST)

            # Convert all back to torch for tensorboard & move to GPU
            ground_truth_tensorboard = torch.from_numpy(ground_truth_tensorboard).to(model.device)
            chunk_tensorboard = torch.from_numpy(chunk_tensorboard).to(model.device)
            output_logits_tensorboard = torch.from_numpy(output_logits_tensorboard).to(model.device)

            # Append to summary lists for tensorboard
            summary_input_scans.append(chunk_tensorboard)
            summary_output_logits.append(output_logits_tensorboard)
            summary_ground_truths.append(ground_truth_tensorboard)

        # Take probability maps and turn into 1 segmentation map, convert to ints
        output_sigmoid = torch.sigmoid(output_logits)
        output_segmentation = torch.where(
            output_sigmoid < 0.5,
            torch.zeros_like(output_sigmoid),
            torch.ones_like(output_sigmoid)).long()

        # Move the prediction to CPU to convert to numpy
        output_segmentation = output_segmentation.cpu().numpy()

        output_segmentation = np.squeeze(output_segmentation)

        # Resize prediction to ground truth annotation size
        output_segmentation = cv2.resize(
            output_segmentation,
            (width, height),
            interpolation=cv2.INTER_NEAREST)

        # Store images, overlays of predictions, ground truths
        if len(visual_paths) > 0:
            chunk = chunk.cpu().numpy()
            output_sigmoid = output_sigmoid[0, 0].cpu().data.numpy()

            # Resize soft prediction to ground truth annotation size
            output_sigmoid = cv2.resize(
                output_sigmoid,
                (width, height),
                interpolation=cv2.INTER_NEAREST)

            assert chunk.shape[0] == 1, ('Batch size should be 1', chunk.shape)

            save_prediction_img(
                chunk=chunk,
                idx=idx,
                chunk_idx=0,
                output_segmentation=output_segmentation,
                output_segmentation_soft=output_sigmoid,
                ground_truth_2d=ground_truth,
                visual_paths=visual_paths
            )

        # Accumulate amount of time for predicting this patient's scan
        patient_prediction_time += chunk_prediction_time

        # Store patient's prediction time in array
        prediction_times[idx] = patient_prediction_time

        # Save output predictions
        if len(output_paths) > 0:
            save_numpy(output_segmentation, output_paths[idx])

        if ground_truths is not None:
            # Compute and store metrics for each patient
            dice_scores[idx] = dice_score(output_segmentation, ground_truth)
            ious[idx] = IOU(output_segmentation, ground_truth)
            precisions[idx] = precision(output_segmentation, ground_truth)
            recalls[idx] = recall(output_segmentation, ground_truth)

            accuracies[idx] = accuracy(output_segmentation, ground_truth)
            specificities[idx] = specificity(output_segmentation, ground_truth)
            f1s[idx] = f1(output_segmentation, ground_truth)
            auc_rocs[idx] = auc_roc(output_segmentation, ground_truth)

            # If saving images, also print out current metrics
            if len(visual_paths) > 0:
                log('{:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>25}'.format(
                    'Patient', 'Dice', 'IOU', 'Precision', 'Recall', 'Smallest-10 Lesion Sizes'), log_path)
                log('{:>9}  {:9.3f}  {:9.3f}  {:9.3f}  {:9.3f}  {}'.format(
                    idx, dice_scores[idx], ious[idx], precisions[idx], recalls[idx], sorted(lesion_sizes)[0:10]), log_path)

            # Calculate per-class metrics
            class_histogram = compute_prediction_hist(ground_truth.flatten(), output_segmentation.flatten(), n_classes)
            per_class_dices[idx] = per_class_dice(class_histogram)
            per_class_ious[idx] = per_class_iou(class_histogram)
            per_class_recalls[idx] = per_class_recall(class_histogram)
            per_class_precisions[idx] = per_class_precision(class_histogram)

    # Calculate average patient prediction time
    mean_patient_prediction_time = np.mean(prediction_times)

    if ground_truths is not None:
        # Calculate mean value of each metric across all scans
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(ious)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        mean_accuracy = np.mean(accuracies)
        mean_specificity = np.mean(specificities)
        mean_f1 = np.mean(f1s)
        mean_auc_roc = np.mean(auc_rocs)

        # tuples of n_classes + 1 elements; means of lesion_class, non_lesion_class, all
        mean_per_class_dice = perclass2mean(per_class_dices)
        mean_per_class_iou = perclass2mean(per_class_ious)
        mean_per_class_precision = perclass2mean(per_class_precisions)
        mean_per_class_recall = perclass2mean(per_class_recalls)

        # Log metrics to tensorboard
        if summary_writer is not None:
            summary_input_scans = torch.stack(summary_input_scans, dim=0)
            summary_output_logits = torch.stack(summary_output_logits, dim=0)
            summary_ground_truths = torch.stack(summary_ground_truths, dim=0)

            model.log_summary(
                input_scan=summary_input_scans,
                output_logits=summary_output_logits,
                ground_truth=summary_ground_truths,
                scalar_dictionary={
                    'eval_metric_dice': mean_dice,
                    'eval_metric_iou': mean_iou,
                    'eval_metric_precision': mean_precision,
                    'eval_metric_recall': mean_recall,
                    'eval_metric_accuracy': mean_accuracy,
                    'eval_metric_specificity': mean_specificity,
                    'eval_metric_f1': mean_f1,
                    'eval_metric_auc_roc': mean_auc_roc
                },
                summary_writer=summary_writer,
                step=step,
                n_display=4
            )
        # Compare to best results
        means = {
            'dice': mean_dice,
            'iou': mean_iou,
            'precision': mean_precision,
            'recall': mean_recall,
            'per_class_dice': mean_per_class_dice,
            'per_class_iou': mean_per_class_iou,
            'per_class_precision': mean_per_class_precision,
            'per_class_recall': mean_per_class_recall,
            'accuracy': mean_accuracy,
            'specificity': mean_specificity,
            'f1': mean_f1,
            'auc_roc': mean_auc_roc
        }
        best_results = update_best_results(best_results, step, means)

        # log current results
        log('Validation results: ', log_path)

        log('Mean patient prediction time: {:.3f} seconds'.format(mean_patient_prediction_time), log_path)

        log('Current step:', log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Step', 'Dice', 'IOU', 'Precision', 'Recall'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            step, mean_dice, mean_iou, mean_precision, mean_recall), log_path)

        log('{:>10}'.format('Per class metrics:'), log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Class', 'dice', 'iou', 'precision', 'recall'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Non-Lesion', mean_per_class_dice[0], mean_per_class_iou[0], mean_per_class_precision[0], mean_per_class_recall[0]), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Lesion', mean_per_class_dice[1], mean_per_class_iou[1], mean_per_class_precision[1], mean_per_class_recall[1]), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Mean', mean_per_class_dice[2], mean_per_class_iou[2], mean_per_class_precision[2], mean_per_class_recall[2]), log_path)
        # RGB specific
        log('RGB Metrics:', log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Step', 'Accuracy', 'Spec.', 'F1', 'AUC_ROC'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            step, mean_accuracy, mean_specificity, mean_f1, mean_auc_roc), log_path)
        # Log best results
        log_best_results(log_path, best_results)

    return best_results

def small_lesion_validate(model,
                          dataloader,
                          transforms,
                          small_lesion_idxs,
                          step,
                          log_path,
                          ground_truths=None,
                          test_time_flip_type='',
                          n_chunk=1,
                          n_classes=2,
                          dataset_means=[0, 0, 0, 0, 0],
                          best_results=None,
                          summary_writer=None,
                          output_paths=[],
                          visual_paths=[]):
    '''
    Given validation data and ground truths and a model, create predictions
        and return (optionally best) results

    Arg(s):
        model : SPiNModel
            Trained model to make prediction
        dataloader : torch.utils.data.DataLoader
            Validation dataloader for scans (NOT shuffled)
        transforms : Transforms
            transformations for the data loader
        small_lesion_idxs : list[set[int]]
            idxs of chunks that have only small lesions. Same order as dataloader
        step : int
            Which step of training (mostly for best_results updates and tensorboard)
        log_path : string
            Path to logging file
        ground_truths : list[numpy[int64]]
            ground truth annotations corresonding to same order as dataloader
        test_time_flip_type : string
            flipping types used at test time
        n_chunk : int
            number of chunks we grab to feed into model per prediction
        n_classes : int
            Number of classes we are predicting (default = 2 in this case)
        dataset_means  : list[int]
            list of mean values for each data modality
        best_results : dictionary or None
            Optional parameter to compare these results to
        summary_writer : tensorboard SummaryWriter or None
            Optional parameter for validation SummaryWriter
        output_paths : list of str
            Output paths, if empty, no nii file is saved
        visual_paths : list of str
            if not empty, visualize and store predictions as png
    Returns:
        best_results : dictionary
            If best_results was not None, return the better of this validate output
            and best_results. Otherwise return these results.
    '''

    n_samples = len(dataloader)

    # Choose a sample to log to tensorboard
    summary_input_scans = []
    summary_output_logits = []
    summary_ground_truths = []

    assert len(dataloader) == len(small_lesion_idxs)

    # Create lists for small lesion segmentations and ground truths
    small_lesion_segmentations = []
    small_lesion_ground_truths = []

    if ground_truths is not None:
        ground_truth_shape = ground_truths[0].shape

        height = ground_truth_shape[0]
        width = ground_truth_shape[1]

    if len(output_paths) > 0:
        assert len(output_paths) == n_samples, (len(output_paths), n_samples)

    # Iterate through each scan
    for idx, (scan, small_lesion_idx_set) in enumerate(zip(dataloader, small_lesion_idxs)):

        if ground_truths is not None:
            ground_truth = ground_truths[idx]
            patient_prediction = np.zeros_like(ground_truth)
        else:
            channel, height, width = scan.shape[1], scan.shape[-2], scan.shape[-1]
            patient_prediction = np.zeros((height, width, channel))

        # Iterate through each chunk of each scan, calculate and store metric
        for chunk_idx in range(scan.shape[1]):

            if chunk_idx not in small_lesion_idx_set:
                continue

            # Scan shape: 1 x C x D x H x W
            chunk = get_n_chunk(
                scan,
                chunk_idx,
                n_chunk,
                constants=dataset_means,
                is_torch=True,
                input_type='BCDHW')

            # Move chunk to same device as model
            chunk = chunk.cuda()

            [chunk] = transforms.transform(
                images_arr=[chunk],
                random_transform_probability=0.0)

            # Forward through super resolution and segmentation model
            output_logits = model.forward(chunk)

            # average predictions for horizontally/vertically flipped images:
            output_logits = average_flip(model, output_logits[-1], chunk, test_time_flip_type)

            # Save scan, segmentation, and ground truth chunk for tensorboard
            if chunk_idx == 90 and summary_writer is not None and ground_truths is not None:
                if np.random.rand() < 0.10 or idx == n_samples - 1:
                    ground_truth_chunk = ground_truth[..., chunk_idx]
                    ground_truth_chunk = cv2.resize(
                        ground_truth_chunk,
                        chunk.shape[-2:][::-1],
                        interpolation=cv2.INTER_NEAREST)
                    ground_truth_chunk = torch.from_numpy(ground_truth_chunk)

                    # Move ground truth to same device as model
                    ground_truth_chunk = ground_truth_chunk.to(model.device)

                    summary_input_scans.append(torch.squeeze(torch.squeeze(chunk, axis=0), axis=1))
                    summary_output_logits.append(torch.squeeze(torch.squeeze(output_logits, axis=0), axis=0))
                    summary_ground_truths.append(ground_truth_chunk)

            # Take probability maps and turn into 1 segmentation map, convert to ints
            output_sigmoid = torch.sigmoid(output_logits)
            output_segmentation = torch.where(
                output_sigmoid < 0.5,
                torch.zeros_like(output_sigmoid),
                torch.ones_like(output_sigmoid)).long()

            # Move the prediction to CPU to convert to numpy
            output_segmentation = output_segmentation.cpu().numpy()

            output_segmentation = np.squeeze(output_segmentation)

            # Resize prediction to ground truth annotation size
            output_segmentation = cv2.resize(
                output_segmentation,
                (width, height),
                interpolation=cv2.INTER_NEAREST)

            # Append prediction and ground truth to lists
            small_lesion_segmentations.append(output_segmentation)
            small_lesion_ground_truths.append(ground_truth[..., chunk_idx])

            if len(visual_paths) > 0:
                chunk = chunk.cpu().numpy()
                output_sigmoid = output_sigmoid[0, 0].cpu().data.numpy()

                # Resize soft prediction to ground truth annotation size
                output_sigmoid = cv2.resize(
                    output_sigmoid,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST)

                assert chunk.shape[0] == 1, ('Batch size should be 1', chunk.shape)
                ground_truth_2d = ground_truth[:, :, chunk_idx]

                save_prediction_img(
                    chunk=chunk,
                    idx=idx,
                    chunk_idx=chunk_idx,
                    output_segmentation=output_segmentation,
                    output_segmentation_soft=output_sigmoid,
                    ground_truth_2d=ground_truth_2d,
                    visual_paths=visual_paths)

        if len(output_paths) > 0:
            # Data type should be unsigned short
            patient_prediction = patient_prediction.astype(np.ushort)

            # save predictions in nii
            save_numpy_to_nii(patient_prediction, output_paths[idx])

    if ground_truths is not None:
        # Stack chunks into 1 tensor
        small_lesion_segmentations = np.stack(small_lesion_segmentations, axis=-1)
        small_lesion_ground_truths = np.stack(small_lesion_ground_truths, axis=-1)

        # Calculate overall metrics for the scans with only small lesions
        metric_dice = dice_score(small_lesion_segmentations, small_lesion_ground_truths)
        metric_iou = IOU(small_lesion_segmentations, small_lesion_ground_truths)
        metric_precision = precision(small_lesion_segmentations, small_lesion_ground_truths)
        metric_recall = recall(small_lesion_segmentations, small_lesion_ground_truths)

        # Calculate per class metrics
        class_histogram = compute_prediction_hist(
            small_lesion_ground_truths.flatten(),
            small_lesion_segmentations.flatten(),
            n_classes)

        metric_per_class_dice = np.expand_dims(per_class_dice(class_histogram), axis=0)
        metric_per_class_iou = np.expand_dims(per_class_iou(class_histogram), axis=0)
        metric_per_class_recall = np.expand_dims(per_class_recall(class_histogram), axis=0)
        metric_per_class_precision = np.expand_dims(per_class_precision(class_histogram), axis=0)

        # tuples of n_classes + 1 elements; means of lesion_class, non_lesion_class, all
        mean_per_class_dice = perclass2mean(metric_per_class_dice)
        mean_per_class_iou = perclass2mean(metric_per_class_iou)
        mean_per_class_precision = perclass2mean(metric_per_class_precision)
        mean_per_class_recall = perclass2mean(metric_per_class_recall)

        # Log metrics to tensorboard
        if summary_writer is not None:
            summary_input_scans = torch.stack(summary_input_scans, dim=0)
            summary_output_logits = torch.stack(summary_output_logits, dim=0)
            summary_ground_truths = torch.stack(summary_ground_truths, dim=0)

            model.log_summary(
                input_scan=summary_input_scans,
                output_logits=summary_output_logits,
                ground_truth=summary_ground_truths,
                scalar_dictionary={
                    'eval_metric_dice': metric_dice,
                    'eval_metric_iou': metric_iou,
                    'eval_metric_precision': metric_precision,
                    'eval_metric_recall': metric_recall
                },
                summary_writer=summary_writer,
                step=step,
                n_display=4)

        # Compare to best results
        if best_results is not None:
            n_improvements = 0
            if np.round(metric_dice, 3) > np.round(best_results['dice'], 3):
                n_improvements += 1
            if np.round(metric_precision, 3) > np.round(best_results['precision'], 3):
                n_improvements += 1
            if np.round(metric_recall, 3) > np.round(best_results['recall'], 3):
                n_improvements += 1

            if n_improvements >= 2:
                best_results['step'] = step
                best_results['dice'] = metric_dice
                best_results['iou'] = metric_iou
                best_results['precision'] = metric_precision
                best_results['recall'] = metric_recall

                best_results['per_class_dice'] = mean_per_class_dice
                best_results['per_class_iou'] = mean_per_class_iou
                best_results['per_class_precision'] = mean_per_class_precision
                best_results['per_class_recall'] = mean_per_class_recall
        # Store results in best_results
        else:
            best_results = {}
            best_results['step'] = step
            best_results['dice'] = metric_dice
            best_results['iou'] = metric_iou
            best_results['precision'] = metric_precision
            best_results['recall'] = metric_recall

            best_results['per_class_dice'] = mean_per_class_dice
            best_results['per_class_iou'] = mean_per_class_iou
            best_results['per_class_precision'] = mean_per_class_precision
            best_results['per_class_recall'] = mean_per_class_recall

        # Store current results as latest
        best_results['step_last'] = step
        best_results['dice_last'] = metric_dice
        best_results['iou_last'] = metric_iou
        best_results['precision_last'] = metric_precision
        best_results['recall_last'] = metric_recall

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
            step, metric_dice, metric_iou, metric_precision, metric_recall), log_path)

        log('{:>10}'.format('Per class metrics:'), log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('Class', 'dice', 'iou', 'precision', 'recall'), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Non-Lesion', mean_per_class_dice[0], mean_per_class_iou[0], mean_per_class_precision[0], mean_per_class_recall[0]), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Lesion', mean_per_class_dice[1], mean_per_class_iou[1], mean_per_class_precision[1], mean_per_class_recall[1]), log_path)
        log('{:>10}  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}'.format(
            'Mean', mean_per_class_dice[2], mean_per_class_iou[2], mean_per_class_precision[2], mean_per_class_recall[2]), log_path)

        # Log best results
        log_best_results(log_path, best_results)

    return best_results

def display_time(seconds, granularity=2):
    intervals = (('days', 86400),    # 60 * 60 * 24
                 ('hours', 3600),    # 60 * 60
                 ('minutes', 60),
                 ('seconds', 1))

    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                # make the word singular:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))

        else:
            # print 0.0 minutes and seconds if that is the case
            if name in ['minutes', 'seconds']:
                result.append("{} {}".format(value, name))

    return ', '.join(result[:granularity])

def update_best_results(best_results, step, means):
    '''
    Compares previous best results to current results and updates if necessary

    Arg(s)
        best_results : dict
            dictionary mapping metrics to values of best step
        step : int
            current step of training
        means : dict
            dictionary mapping the metrics to current mean values (including per-class values)

    Returns:
        best_results : dictionary
            If best_results was not None, return the better of this validate output
            and best_results. Otherwise return these results.
    '''
    hasRGBMetrics = 'accuracy' in means
    # Extract mean values
    mean_dice = means['dice']
    mean_iou = means['iou']
    mean_precision = means['precision']
    mean_recall = means['recall']

    # Extract per class mean values
    mean_per_class_dice = means['per_class_dice']
    mean_per_class_iou = means['per_class_iou']
    mean_per_class_precision = means['per_class_precision']
    mean_per_class_recall = means['per_class_recall']

    if hasRGBMetrics:
        mean_accuracy = means['accuracy']
        mean_specificity = means['specificity']
        mean_f1 = means['f1']
        mean_auc_roc = means['auc_roc']

    # Compare to best results
    if best_results is not None:
        n_improvements = 0
        if np.round(mean_dice, 3) > np.round(best_results['dice'], 3):
            n_improvements += 1
        if np.round(mean_precision, 3) > np.round(best_results['precision'], 3):
            n_improvements += 1
        if np.round(mean_recall, 3) > np.round(best_results['recall'], 3):
            n_improvements += 1

        if n_improvements >= 2:
            best_results['step'] = step
            best_results['dice'] = mean_dice
            best_results['iou'] = mean_iou
            best_results['precision'] = mean_precision
            best_results['recall'] = mean_recall

            best_results['per_class_dice'] = mean_per_class_dice
            best_results['per_class_iou'] = mean_per_class_iou
            best_results['per_class_precision'] = mean_per_class_precision
            best_results['per_class_recall'] = mean_per_class_recall

            if hasRGBMetrics:
                best_results['accuracy'] = mean_accuracy
                best_results['specificity'] = mean_specificity
                best_results['f1'] = mean_f1
                best_results['auc_roc'] = mean_auc_roc
    # Store results in best_results
    else:
        best_results = {}
        best_results['step'] = step
        best_results['dice'] = mean_dice
        best_results['iou'] = mean_iou
        best_results['precision'] = mean_precision
        best_results['recall'] = mean_recall

        best_results['per_class_dice'] = mean_per_class_dice
        best_results['per_class_iou'] = mean_per_class_iou
        best_results['per_class_precision'] = mean_per_class_precision
        best_results['per_class_recall'] = mean_per_class_recall

        if hasRGBMetrics:
            best_results['accuracy'] = mean_accuracy
            best_results['specificity'] = mean_specificity
            best_results['f1'] = mean_f1
            best_results['auc_roc'] = mean_auc_roc

    # Store current results as latest
    best_results['step_last'] = step
    best_results['dice_last'] = mean_dice
    best_results['iou_last'] = mean_iou
    best_results['precision_last'] = mean_precision
    best_results['recall_last'] = mean_recall

    best_results['per_class_dice_last'] = mean_per_class_dice
    best_results['per_class_iou_last'] = mean_per_class_iou
    best_results['per_class_precision_last'] = mean_per_class_precision
    best_results['per_class_recall_last'] = mean_per_class_recall

    if hasRGBMetrics:
        best_results['accuracy_last'] = mean_accuracy
        best_results['specificity_last'] = mean_specificity
        best_results['f1_last'] = mean_f1
        best_results['auc_roc_last'] = mean_auc_roc

    return best_results