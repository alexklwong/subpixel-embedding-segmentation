import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import data_utils, datasets
import global_constants as settings
from eval_utils import validateMRI, validateRGB, small_lesion_validate
from log_utils import log, save_MRI_prediction_img, save_RGB_prediction_img
from data_utils import get_scan_type
from transforms import Transforms
from spin_model import SPiNModel
from sampler import PositiveClassSampler


def train(train_multimodal_scan_paths,
          train_ground_truth_path,
          val_multimodal_scan_paths,
          val_ground_truth_path,
          # Input settings
          n_chunk=settings.N_CHUNK,
          n_height=settings.O_HEIGHT,
          n_width=settings.O_WIDTH,
          # Normalization settings
          dataset_normalization='standard',
          dataset_means=[settings.ATLAS_MEAN],
          dataset_stddevs=[settings.ATLAS_SD],
          # Subpixel embedding network settings
          encoder_type_subpixel_embedding=settings.ENCODER_TYPE_SUBPIXEL_EMBEDDING,
          n_filters_encoder_subpixel_embedding=settings.N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING,
          decoder_type_subpixel_embedding=settings.DECODER_TYPE_SUBPIXEL_EMBEDDING,
          n_filter_decoder_subpixel_embedding=settings.N_FILTER_DECODER_SUBPIXEL_EMBEDDING,
          output_channels_subpixel_embedding=settings.OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING,
          output_func_subpixel_embedding=settings.OUTPUT_FUNC,
          # Segmentation network settings
          encoder_type_segmentation=settings.ENCODER_TYPE_SEGMENTATION,
          n_filters_encoder_segmentation=settings.N_FILTERS_ENCODER_SEGMENTATION,
          resolutions_subpixel_guidance=settings.RESOLUTIONS_SUBPIXEL_GUIDANCE,
          n_filters_subpixel_guidance=settings.N_FILTERS_SUBPIXEL_GUIDANCE,
          n_convolutions_subpixel_guidance=settings.N_CONVOLUTIONS_SUBPIXEL_GUIDANCE,
          decoder_type_segmentation=settings.DECODER_TYPE_SEGMENTATION,
          n_filters_decoder_segmentation=settings.N_FILTERS_DECODER_SEGMENTATION,
          n_filters_learnable_downsampler=settings.N_FILTERS_LEARNABLE_DOWNSAMPLER,
          kernel_sizes_learnable_downsampler=settings.KERNEL_SIZES_LEARNABLE_DOWNSAMPLER,
          # Weights settings
          weight_initializer=settings.WEIGHT_INITIALIZER,
          activation_func=settings.ACTIVATION_FUNC,
          use_batch_norm=settings.USE_BATCH_NORM,
          # Training settings
          n_batch=settings.N_BATCH,
          learning_rates=settings.LEARNING_RATES,
          learning_schedule=settings.LEARNING_SCHEDULE,
          # Data sampler
          positive_class_sample_rates=[0.95],
          positive_class_sample_schedule=[-1],
          positive_class_size_thresholds=[0],
          # Data augmentation
          augmentation_probabilities=[0.50],
          augmentation_schedule=[-1],
          augmentation_flip_type=['horizontal', 'vertical'],
          augmentation_rotate=30.0,
          augmentation_noise_type='gaussian',
          augmentation_noise_spread=1e-2,
          augmentation_resize_and_pad=[1.0, 1.1],
          # Subpixel embedding loss function
          w_weight_decay_subpixel_embedding=settings.W_WEIGHT_DECAY,
          # Segmentation loss function
          loss_func_segmentation=settings.LOSS_FUNC_SEGMENTATION,
          w_weight_decay_segmentation=settings.W_WEIGHT_DECAY,
          w_positive_class=settings.W_POSITIVE_CLASS,
          # Checkpoint settings
          n_summary=settings.N_SUMMARY,
          n_checkpoint=settings.N_CHECKPOINT,
          checkpoint_path=settings.CHECKPOINT_PATH,
          restore_path=settings.RESTORE_PATH,
          # Hardware settings
          device=settings.DEVICE,
          n_thread=settings.N_THREAD):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_path = os.path.join(checkpoint_path, 'model-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    log('Training input path(s):', log_path)
    for path in train_multimodal_scan_paths:
        log(path, log_path)
    log('Training ground truth annotation path:', log_path)
    log(train_ground_truth_path, log_path)

    log('Validation input path(s):', log_path)
    for path in val_multimodal_scan_paths:
        log(path, log_path)
    log('Validation ground truth annotation path:', log_path)
    log(val_ground_truth_path, log_path)
    log('', log_path)

    # Read paths for training
    train_multimodal_scan_paths = [
        data_utils.read_paths(path) for path in train_multimodal_scan_paths
    ]
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)

    for paths in train_multimodal_scan_paths:
        assert len(paths) == len(train_ground_truth_paths)

    # Read paths for validation
    val_multimodal_scan_paths = [
        data_utils.read_paths(path) for path in val_multimodal_scan_paths
    ]
    ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    for paths in val_multimodal_scan_paths:
        assert len(paths) == len(ground_truth_paths)

    n_train_sample = len(train_multimodal_scan_paths[0])
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    # Load ground truth scans
    val_ground_truths = []
    for path in ground_truth_paths:
        ground_truth = np.where(np.load(path) > 0, 1, 0)
        val_ground_truths.append(ground_truth)

    positive_class_sampler = PositiveClassSampler(
        positive_class_sample_rates=positive_class_sample_rates,
        positive_class_sample_schedule=positive_class_sample_schedule,
        positive_class_size_thresholds=positive_class_size_thresholds)

    # Determine which data type to use
    scan_type = get_scan_type(train_multimodal_scan_paths[0][0])
    # Training dataloader
    if scan_type == 'MRI':
        train_dataloader = torch.utils.data.DataLoader(
            datasets.SPiNMRITrainingDataset(
                multimodal_scan_paths=train_multimodal_scan_paths,
                ground_truth_paths=train_ground_truth_paths,
                shape=(n_chunk, n_height, n_width),
                padding_constants=dataset_means,
                positive_class_sampler=positive_class_sampler),
            batch_size=n_batch,
            shuffle=True,
            num_workers=n_thread,
            drop_last=False)

        # Validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            datasets.SPiNMRIInferenceDataset(
                multimodal_scan_paths=val_multimodal_scan_paths,
                shape=(None, None)),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        val_transforms = Transforms(
            dataset_means=dataset_means,
            dataset_stddevs=dataset_stddevs,
            dataset_normalization=dataset_normalization)

        input_channels = n_chunk
        # Set appropriate functions
        validate = validateMRI
        save_prediction_img = save_MRI_prediction_img
    elif scan_type == 'RGB':
        train_dataloader = torch.utils.data.DataLoader(
            datasets.SPiNRGBTrainingDataset(
                scan_paths=train_multimodal_scan_paths[0],
                ground_truth_paths=train_ground_truth_paths,
                shape=(n_height, n_width),
                padding_constant=dataset_means[0]),
            batch_size=n_batch,
            shuffle=True,
            num_workers=n_thread,
            drop_last=False)

        # Validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            datasets.SPiNRGBInferenceDataset(
                scan_paths=val_multimodal_scan_paths[0],
                shape=(None, None)),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        input_channels = 3  # RGB scans have 3 channels
        validate = validateRGB
        save_prediction_img = save_RGB_prediction_img
    else:
        raise ValueError('Dataset not supported.')

    train_transforms = Transforms(
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs,
        dataset_normalization=dataset_normalization,
        random_flip_type=augmentation_flip_type,
        random_rotate=augmentation_rotate,
        random_noise_type=augmentation_noise_type,
        random_noise_spread=augmentation_noise_spread,
        random_resize_and_pad=augmentation_resize_and_pad)

    val_transforms = Transforms(
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs,
        dataset_normalization=dataset_normalization)

    # Build SubPixel Network (SPiN)
    model = SPiNModel(
        input_channels=input_channels,
        encoder_type_subpixel_embedding=encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=decoder_type_subpixel_embedding,
        output_channels_subpixel_embedding=output_channels_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=n_filter_decoder_subpixel_embedding,
        output_func_subpixel_embedding=output_func_subpixel_embedding,
        encoder_type_segmentation=encoder_type_segmentation,
        n_filters_encoder_segmentation=n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=n_convolutions_subpixel_guidance,
        decoder_type_segmentation=decoder_type_segmentation,
        n_filters_decoder_segmentation=n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
        device=device)

    _, parameters_subpixel_embedding, parameters_segmentation = model.parameters()

    if restore_path is not None:
        model.restore_model(restore_path)

    train_summary = SummaryWriter(event_path + '-train')
    val_summary = SummaryWriter(event_path + '-val')

    # Variable to store best validation results
    best_results = None

    # Log settings used for training
    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_chunk=n_chunk,
        n_height=n_height,
        n_width=n_width,
        # Normalization settings
        dataset_normalization=dataset_normalization,
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs)

    log_network_settings(
        log_path,
        # Subpixel embedding network settings
        encoder_type_subpixel_embedding=encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=decoder_type_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=n_filter_decoder_subpixel_embedding,
        output_channels_subpixel_embedding=output_channels_subpixel_embedding,
        output_func_subpixel_embedding=output_func_subpixel_embedding,
        # Segmentation network settings
        encoder_type_segmentation=encoder_type_segmentation,
        n_filters_encoder_segmentation=n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=n_convolutions_subpixel_guidance,
        decoder_type_segmentation=decoder_type_segmentation,
        n_filters_decoder_segmentation=n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler,
        # Weights settings
        parameters_subpixel_embedding=parameters_subpixel_embedding,
        parameters_segmentation=parameters_segmentation,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm)

    log_loss_func_settings(
        log_path,
        # Subpixel embedding loss settings
        w_weight_decay_subpixel_embedding=w_weight_decay_subpixel_embedding if 'none' not in encoder_type_subpixel_embedding else None,
        # Segmentation network loss settings
        loss_func_segmentation=loss_func_segmentation,
        w_weight_decay_segmentation=w_weight_decay_segmentation,
        w_positive_class=w_positive_class)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Data sampler
        positive_class_sample_rates=positive_class_sample_rates,
        positive_class_sample_schedule=positive_class_sample_schedule,
        positive_class_size_thresholds=positive_class_size_thresholds,
        # Data augmentation
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_flip_type=augmentation_flip_type,
        augmentation_rotate=augmentation_rotate,
        augmentation_noise_type=augmentation_noise_type,
        augmentation_noise_spread=augmentation_noise_spread,
        augmentation_resize_and_pad=augmentation_resize_and_pad)

    log_system_settings(
        log_path,
        # Checkpoint settings
        n_summary=n_summary,
        n_checkpoint=n_checkpoint,
        checkpoint_path=checkpoint_path,
        restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    # Set up training settings
    train_step = 0
    time_start = time.time()
    log('Begin training...', log_path)

    learning_schedule_pos = 0
    augmentation_schedule_pos = 0
    n_epoch = learning_schedule[-1]

    # Start training
    model.train()

    for epoch in range(1, n_epoch + 1):
        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1

        learning_rate = learning_rates[learning_schedule_pos]

        if 'none' in encoder_type_subpixel_embedding and 'none' in decoder_type_subpixel_embedding:
            optimizer = torch.optim.Adam([
                {
                    'params' : parameters_segmentation,
                    'weight_decay' : w_weight_decay_segmentation
                }],
                lr=learning_rate)
        else:
            optimizer = torch.optim.Adam([
                {
                    'params' : parameters_subpixel_embedding,
                    'weight_decay' : w_weight_decay_subpixel_embedding
                },
                {
                    'params' : parameters_segmentation,
                    'weight_decay' : w_weight_decay_segmentation
                }],
                lr=learning_rate)

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1

        augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        # Update epoch # for sampler
        if positive_class_sampler is not None:
            positive_class_sampler.update_epoch()

        for train_scan, train_ground_truth in train_dataloader:
            train_step = train_step + 1

            # Move data to device
            train_scan = train_scan.to(device)
            train_ground_truth = train_ground_truth.to(device)

            [train_scan], [train_ground_truth] = train_transforms.transform(
                images_arr=[train_scan],
                labels_arr=[train_ground_truth],
                random_transform_probability=augmentation_probability)

            # Forward through the segmentation network
            output_logits = model.forward(train_scan)

            # Compute loss function
            loss = model.compute_loss(
                output_logits=output_logits,
                ground_truth=train_ground_truth,
                loss_func_segmentation=loss_func_segmentation,
                w_positive_class=w_positive_class)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                model.log_summary(
                    input_scan=model.input_scan,
                    output_logits=model.output_logits,
                    ground_truth=model.ground_truth,
                    scalar_dictionary={'loss': model.loss_segmentation},
                    summary_writer=train_summary,
                    step=train_step,
                    n_display=4)

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain), log_path)

                # Save checkpoints
                model.save_model(
                    model_path.format(train_step),
                    train_step,
                    optimizer)

                # Switch to eval mode and perform validation
                model.eval()

                with torch.no_grad():
                    best_results = validate(
                        model=model,
                        dataloader=val_dataloader,
                        transforms=val_transforms,
                        save_prediction_img=save_prediction_img,
                        ground_truths=val_ground_truths,
                        step=train_step,
                        log_path=log_path,
                        n_chunk=n_chunk,
                        dataset_means=dataset_means,
                        best_results=best_results,
                        summary_writer=val_summary)

                # Switch back to training mode
                model.train()

    # Log last step
    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        n_train_step, n_train_step, loss.item(), time_elapse, time_remain), log_path)

    # Save checkpoints
    model.save_model(
        model_path.format(train_step),
        train_step,
        optimizer)

    # Switch to eval mode and perform validation for last step
    model.eval()

    with torch.no_grad():
        best_results = validate(
            model=model,
            dataloader=val_dataloader,
            transforms=val_transforms,
            save_prediction_img=save_prediction_img,
            ground_truths=val_ground_truths,
            step=train_step,
            log_path=log_path,
            n_chunk=n_chunk,
            dataset_means=dataset_means,
            best_results=best_results,
            summary_writer=val_summary)

    # Close summaries
    train_summary.close()
    val_summary.close()

    return best_results

def run(multimodal_scan_paths,
        ground_truth_path=None,
        small_lesion_idxs_path=None,
        # Input settings
        n_chunk=settings.N_CHUNK,
        # Normalization setting
        dataset_normalization='standard',
        dataset_means=[settings.ATLAS_MEAN],
        dataset_stddevs=[settings.ATLAS_SD],
        # Subpixel embedding network settings
        encoder_type_subpixel_embedding=settings.ENCODER_TYPE_SUBPIXEL_EMBEDDING,
        n_filters_encoder_subpixel_embedding=settings.N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING,
        decoder_type_subpixel_embedding=settings.DECODER_TYPE_SUBPIXEL_EMBEDDING,
        n_filter_decoder_subpixel_embedding=settings.N_FILTER_DECODER_SUBPIXEL_EMBEDDING,
        output_channels_subpixel_embedding=settings.OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING,
        output_func_subpixel_embedding=settings.OUTPUT_FUNC,
        # Segmentation network settings
        encoder_type_segmentation=settings.ENCODER_TYPE_SEGMENTATION,
        n_filters_encoder_segmentation=settings.N_FILTERS_ENCODER_SEGMENTATION,
        resolutions_subpixel_guidance=settings.RESOLUTIONS_SUBPIXEL_GUIDANCE,
        n_filters_subpixel_guidance=settings.N_FILTERS_SUBPIXEL_GUIDANCE,
        n_convolutions_subpixel_guidance=settings.N_CONVOLUTIONS_SUBPIXEL_GUIDANCE,
        decoder_type_segmentation=settings.DECODER_TYPE_SEGMENTATION,
        n_filters_decoder_segmentation=settings.N_FILTERS_DECODER_SEGMENTATION,
        n_filters_learnable_downsampler=settings.N_FILTERS_LEARNABLE_DOWNSAMPLER,
        kernel_sizes_learnable_downsampler=settings.KERNEL_SIZES_LEARNABLE_DOWNSAMPLER,
        # Weights settings
        weight_initializer=settings.WEIGHT_INITIALIZER,
        activation_func=settings.ACTIVATION_FUNC,
        use_batch_norm=settings.USE_BATCH_NORM,
        # Test time augmentation
        augmentation_flip_type=['none'],
        # Checkpoint settings
        checkpoint_path=settings.CHECKPOINT_PATH,
        restore_path=settings.RESTORE_PATH,
        do_visualize_predictions=False,
        # Hardware settings
        device=settings.DEVICE,
        n_thread=settings.N_THREAD):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    log_path = os.path.join(checkpoint_path, 'results.txt')

    if do_visualize_predictions:
        # Get input modality names for ....:
        visual_paths = [
            os.path.join(checkpoint_path, path.split('/')[-1].replace('.txt', ''))
            for path in multimodal_scan_paths
        ]
        for visual_path in visual_paths:
            if not os.path.exists(visual_path):
                os.makedirs(visual_path)
    else:
        visual_paths = []

    log('Input paths:', log_path)
    for path in multimodal_scan_paths:
        log(path, log_path)

    # Read paths for evaluation
    multimodal_scan_paths = [
        data_utils.read_paths(path) for path in multimodal_scan_paths
    ]

    if ground_truth_path is not None:
        log('Ground truth path:', log_path)
        log(ground_truth_path, log_path)

        ground_truth_paths = data_utils.read_paths(ground_truth_path)

        for paths in multimodal_scan_paths:
            assert len(paths) == len(ground_truth_paths)

        # Load ground truth
        ground_truths = []
        for path in ground_truth_paths:
            ground_truth = np.where(np.load(path) > 0, 1, 0)
            ground_truths.append(ground_truth)

    if small_lesion_idxs_path is not None:
        log('Small lesion indices path:', log_path)
        log(small_lesion_idxs_path, log_path)

    log('', log_path)

    # Set up dataloader
     # Determine which data type to use
    scan_type = get_scan_type(multimodal_scan_paths[0][0])
    # Training dataloader
    if scan_type == 'MRI':
        dataloader = torch.utils.data.DataLoader(
            datasets.SPiNMRIInferenceDataset(
                multimodal_scan_paths=multimodal_scan_paths,
                shape=(None, None)),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        input_channels = n_chunk
        validate = validateMRI
        save_prediction_img = save_MRI_prediction_img
    elif scan_type == 'RGB':
        dataloader = torch.utils.data.DataLoader(
            datasets.SPiNRGBInferenceDataset(
                scan_paths=multimodal_scan_paths[0],
                shape=(None, None)),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        input_channels = 3  # RGB
        validate = validateRGB
        save_prediction_img = save_RGB_prediction_img

    transforms = Transforms(
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs,
        dataset_normalization=dataset_normalization)

    # Obtain indices of scans with small lesions
    if small_lesion_idxs_path is not None and small_lesion_idxs_path != '':
        small_lesion_idxs = data_utils.parse_small_lesion_txt(
            small_lesion_idxs_path,
            n_samples=len(dataloader))
    else:
        small_lesion_idxs = None

    # Build subpixel network (SPiN)
    model = SPiNModel(
        input_channels=input_channels,
        encoder_type_subpixel_embedding=encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=decoder_type_subpixel_embedding,
        output_channels_subpixel_embedding=output_channels_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=n_filter_decoder_subpixel_embedding,
        output_func_subpixel_embedding=output_func_subpixel_embedding,
        encoder_type_segmentation=encoder_type_segmentation,
        n_filters_encoder_segmentation=n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=n_convolutions_subpixel_guidance,
        decoder_type_segmentation=decoder_type_segmentation,
        n_filters_decoder_segmentation=n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
        device=device)

    _, parameters_subpixel_embedding, parameters_segmentation = model.parameters()

    if restore_path is not None:

        assert os.path.isfile(restore_path), \
            'Cannot find retore path: {}'.format(restore_path)

        model.restore_model(restore_path)

    # Log run settings
    log_input_settings(
        log_path,
        # Batch settings
        n_chunk=n_chunk,
        # Normalization settings
        dataset_normalization=dataset_normalization,
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs)

    log_network_settings(
        log_path,
        # Subpixel embedding network settings
        encoder_type_subpixel_embedding=encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=decoder_type_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=n_filter_decoder_subpixel_embedding,
        output_channels_subpixel_embedding=output_channels_subpixel_embedding,
        output_func_subpixel_embedding=output_func_subpixel_embedding,
        # Segmentation network settings
        encoder_type_segmentation=encoder_type_segmentation,
        n_filters_encoder_segmentation=n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=n_convolutions_subpixel_guidance,
        decoder_type_segmentation=decoder_type_segmentation,
        n_filters_decoder_segmentation=n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler,
        # Weights settings
        parameters_subpixel_embedding=parameters_subpixel_embedding,
        parameters_segmentation=parameters_segmentation,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    log('Begin evaluation...', log_path)
    with torch.no_grad():

        model.eval()
        best_results = None

        if ground_truth_path is not None:

            if small_lesion_idxs is not None:
                best_results = small_lesion_validate(
                    model=model,
                    dataloader=dataloader,
                    transforms=transforms,
                    small_lesion_idxs=small_lesion_idxs,
                    ground_truths=ground_truths,
                    test_time_flip_type=augmentation_flip_type,
                    step=0,
                    log_path=log_path,
                    dataset_means=dataset_means,
                    n_chunk=n_chunk,
                    best_results=best_results,
                    visual_paths=visual_paths)
            else:
                best_results = validate(
                    model=model,
                    dataloader=dataloader,
                    transforms=transforms,
                    step=0,
                    log_path=log_path,
                    save_prediction_img=save_prediction_img,
                    ground_truths=ground_truths,
                    test_time_flip_type=augmentation_flip_type,
                    dataset_means=dataset_means,
                    n_chunk=n_chunk,
                    best_results=best_results,
                    visual_paths=visual_paths)
        else:
            # Run without ground truth, will only save results
            assert len(visual_paths) > 0, \
                'No ground truth nor checkpoint output path is provided, so no evaluation or storing will be done.'

            validate(
                model=model,
                dataloader=dataloader,
                transforms=transforms,
                log_path=log_path,
                save_prediction_img=save_prediction_img,
                test_time_flip_type=augmentation_flip_type,
                step=0,
                dataset_means=dataset_means,
                n_chunk=n_chunk,
                best_results=best_results,
                visual_paths=visual_paths)

    return best_results


'''
Logging helper functions
'''
def log_input_settings(log_path,
                       n_batch=None,
                       n_chunk=None,
                       n_height=None,
                       n_width=None,
                       dataset_normalization='standard',
                       dataset_means=[0],
                       dataset_stddevs=[1]):

    input_settings_text = ''
    input_settings_vars = []

    if n_batch is not None:
        input_settings_text = input_settings_text + 'n_batch={}'
        input_settings_vars.append(n_batch)

    input_settings_text = \
        input_settings_text + '  ' if len(input_settings_text) > 0 else input_settings_text

    if n_chunk is not None:
        input_settings_text = input_settings_text + 'n_chunk={}'
        input_settings_vars.append(n_chunk)

    input_settings_text = \
        input_settings_text + '  ' if len(input_settings_text) > 0 else input_settings_text

    if n_height is not None:
        input_settings_text = input_settings_text + 'n_height={}'
        input_settings_vars.append(n_height)

    input_settings_text = \
        input_settings_text + '  ' if len(input_settings_text) > 0 else input_settings_text

    if n_width is not None:
        input_settings_text = input_settings_text + 'n_width={}'
        input_settings_vars.append(n_width)

    log('Input settings:', log_path)
    log(input_settings_text.format(*input_settings_vars),
        log_path)
    log('', log_path)

    log('Normalization settings:', log_path)
    log('dataset_normalization={}'.format(dataset_normalization),
        log_path)
    log('dataset_means={}'.format(dataset_means),
        log_path)
    log('dataset_stddevs={}'.format(dataset_stddevs),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Subpixel embedding network settings
                         encoder_type_subpixel_embedding,
                         n_filters_encoder_subpixel_embedding,
                         decoder_type_subpixel_embedding,
                         n_filter_decoder_subpixel_embedding,
                         output_channels_subpixel_embedding,
                         output_func_subpixel_embedding,
                         # Segmentation network settings
                         encoder_type_segmentation,
                         n_filters_encoder_segmentation,
                         resolutions_subpixel_guidance,
                         n_filters_subpixel_guidance,
                         n_convolutions_subpixel_guidance,
                         decoder_type_segmentation,
                         n_filters_decoder_segmentation,
                         n_filters_learnable_downsampler,
                         kernel_sizes_learnable_downsampler,
                         # Weights settings
                         parameters_subpixel_embedding,
                         parameters_segmentation,
                         weight_initializer,
                         activation_func,
                         use_batch_norm):

    log('Subpixel embedding network settings:', log_path)
    if 'none' in encoder_type_subpixel_embedding and 'none' in decoder_type_subpixel_embedding:
        log('N/A', log_path)
    else:
        log('encoder_type={}'.format(encoder_type_subpixel_embedding),
            log_path)
        log('n_filters_encoder={}'.format(n_filters_encoder_subpixel_embedding),
            log_path)
        log('decoder_type={}'.format(decoder_type_subpixel_embedding),
            log_path)
        log('n_filter_decoder={}  output_channels={}'.format(
            n_filter_decoder_subpixel_embedding, output_channels_subpixel_embedding),
            log_path)
        log('output_func={}'.format(output_func_subpixel_embedding),
            log_path)
        log('Subpixel guidance settings:', log_path)
        log('resolutions={}'.format(resolutions_subpixel_guidance),
            log_path)
        log('n_filters={}'.format(n_filters_subpixel_guidance),
            log_path)
        log('n_convolutions={}'.format(n_convolutions_subpixel_guidance),
            log_path)
        log('', log_path)

    log('Segmentation network settings:', log_path)
    log('encoder_type={}'.format(encoder_type_segmentation),
        log_path)
    log('n_filters_encoder={}'.format(n_filters_encoder_segmentation),
        log_path)
    log('decoder_type={}'.format(decoder_type_segmentation),
        log_path)
    log('n_filters_decoder={}'.format(n_filters_decoder_segmentation),
        log_path)
    log('', log_path)

    log('Learnable downsampler settings:', log_path)
    log('use_learnable_downsampler={}'.format(
        'learnable_downsampler' in decoder_type_segmentation),
        log_path)
    log('n_filters_learnable_downsampler={}'.format(
        n_filters_learnable_downsampler),
        log_path)
    log('kernel_sizes_learnable_downsampler={}'.format(
        kernel_sizes_learnable_downsampler),
        log_path)
    log('', log_path)

    n_parameter_subpixel_embedding = sum(p.numel() for p in parameters_subpixel_embedding)
    n_parameter_segmentation = sum(p.numel() for p in parameters_segmentation)

    n_parameter = n_parameter_segmentation + n_parameter_subpixel_embedding

    log('Weight settings:', log_path)
    log('Parameters: n_parameter={}  n_parameter_segmentation={}  n_parameter_subpixel_embedding={}'.format(
        n_parameter, n_parameter_segmentation, n_parameter_subpixel_embedding),
        log_path)
    log('weight_initializer={}  activation_func={}  use_batch_norm={}'.format(
        weight_initializer, activation_func, use_batch_norm),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           w_weight_decay_subpixel_embedding=None,
                           loss_func_segmentation='cross_entropy',
                           w_weight_decay_segmentation=0.00,
                           w_positive_class=1.00):

    log('Subpixel embedding loss function settings:', log_path)

    if w_weight_decay_subpixel_embedding is None:
        log('N/A', log_path)
    else:
        log('w_weight_decay={:.1e}'.format(w_weight_decay_subpixel_embedding),
            log_path)
        log('', log_path)

    log('Segmentation loss function settings:', log_path)
    log('loss_func={}'.format(loss_func_segmentation),
        log_path)
    log('w_weight_decay={:.1e}  w_positive_class={}'.format(
        w_weight_decay_segmentation, w_positive_class),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates=[3e-4, 1e-4, 5e-5],
                          learning_schedule=[400, 1400, 1600],
                          # Data sampler
                          positive_class_sample_rates=[0.95],
                          positive_class_sample_schedule=[-1],
                          positive_class_size_thresholds=[0],
                          # Data augmentation
                          augmentation_probabilities=[0.50],
                          augmentation_schedule=[-1],
                          augmentation_flip_type=['horizontal', 'vertical'],
                          augmentation_rotate=30.0,
                          augmentation_noise_type='gaussian',
                          augmentation_noise_spread=1e-2,
                          augmentation_resize_and_pad=[1.0, 1.1]):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('Schedule format: [epoch (step) : value]', log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} ({}-{}) : {}'.format(
            start, end, start * (n_train_sample // n_batch), end * (n_train_sample // n_batch), v)
            for start, end, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    if -1 in positive_class_sample_schedule:
        end = learning_schedule[-1]

        log('positive_class_sample_rate_schedule=[%s]' %
            ', '.join('{}-{} ({}-{}) : {}'.format(
                start, end, start * (n_train_sample // n_batch), end * (n_train_sample // n_batch), v)
                for start, v in zip([0] + positive_class_sample_schedule[:-1], positive_class_sample_rates)),
            log_path)
    else:
        log('positive_class_sample_rate_schedule=[%s]' %
            ', '.join('{}-{} ({}-{}) : {}'.format(
                start, end, start * (n_train_sample // n_batch), end * (n_train_sample // n_batch), v)
                for start, end, v in zip([0] + positive_class_sample_schedule[:-1], positive_class_sample_schedule, positive_class_sample_rates)),
            log_path)

    log('positive_class_size_thresholds={}'.format(positive_class_size_thresholds),
        log_path)

    if -1 in augmentation_schedule:
        end = learning_schedule[-1]

        log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} ({}-{}) : {}'.format(
            start, end, start * (n_train_sample // n_batch), end * (n_train_sample // n_batch), v)
            for start, v, in zip([0] + augmentation_schedule[:-1], augmentation_probabilities)),
            log_path)
    else:
        log('augmentation_schedule=[%s]' %
            ', '.join('{}-{} ({}-{}) : {}'.format(
                start, end, start * (n_train_sample // n_batch), end * (n_train_sample // n_batch), v)
                for start, end, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
            log_path)
    log('augmentation_flip_type={}'.format(augmentation_flip_type),
        log_path)
    log('augmentation_rotate={}'.format(augmentation_rotate),
        log_path)
    log('augmentation_noise_type={}  augmentation_noise_spread={}'.format(
        augmentation_noise_type, augmentation_noise_spread),
        log_path)
    log('augmentation_resize_and_pad={}'.format(augmentation_resize_and_pad),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        n_summary=None,
                        n_checkpoint=None,
                        checkpoint_path=None,
                        restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        checkpoint_settings_text = ''
        checkpoint_settings_vars = []

        if n_checkpoint is not None:
            checkpoint_settings_text = checkpoint_settings_text + 'checkpoint_save_frequency={}'
            checkpoint_settings_vars.append(n_checkpoint)

        checkpoint_settings_text = \
            checkpoint_settings_text + '  ' if len(checkpoint_settings_text) > 0 else checkpoint_settings_text

        if n_summary is not None:
            checkpoint_settings_text = checkpoint_settings_text + 'tensorboard_summary_frequency={}'
            checkpoint_settings_vars.append(n_summary)

        if len(checkpoint_settings_text) > 0:
            log(checkpoint_settings_text.format(*checkpoint_settings_vars), log_path)

    if restore_path is not None and restore_path != '':
        log('restore_path={}'.format(restore_path),
            log_path)
    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
