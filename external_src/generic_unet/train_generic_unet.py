import argparse
import os
from generic_unet_main import train
import generic_unet_global_constants as settings

parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_multimodal_scan_paths',
    nargs='+', type=str, required=True, help='Paths to list of training MRI scan paths')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Path to list of ground truth annotation paths')
parser.add_argument('--val_multimodal_scan_paths',
    nargs='+', type=str, required=True, help='Paths to list of validation MRI scan paths')
parser.add_argument('--val_ground_truth_path',
    type=str, required=True, help='Path to list of validation ground truth annotation paths')
# Batch settings
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_chunk',
    type=int, default=settings.N_CHUNK, help='Number of chunks or channels to process at a time')
parser.add_argument('--n_height',
    type=int, default=settings.O_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.O_WIDTH, help='Width of each sample')
# Normalization settings
parser.add_argument('--dataset_normalization',
    type=str, default='standard', help='Type of normalization: none, standard')
parser.add_argument('--dataset_means',
    nargs='+', type=float, default=[settings.ATLAS_MEAN], help='List of mean values of each modality in the dataset. Used after log preprocessing')
parser.add_argument('--dataset_stddevs',
    nargs='+', type=float, default=[settings.ATLAS_SD], help='List of standard deviations of each modality in the dataset. Used after log preprocessing')
# Segmentation network settings
parser.add_argument('--encoder_type_segmentation',
    nargs='+', type=str, default=settings.ENCODER_TYPE_SEGMENTATION, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE_SEGMENTATION)
parser.add_argument('--n_filters_encoder_segmentation',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_SEGMENTATION, help='Number of filters to use in each encoder block')
parser.add_argument('--decoder_type_segmentation',
    nargs='+', type=str, default=settings.DECODER_TYPE_SEGMENTATION, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE_SEGMENTATION)
parser.add_argument('--n_filters_decoder_segmentation',
    nargs='+', type=int, default=settings.N_FILTERS_DECODER_SEGMENTATION, help='Number of filters to use in each decoder block')
parser.add_argument('--weight_initializer',
    type=str, default=settings.WEIGHT_INITIALIZER, help='Weight initializers: %s' % settings.WEIGHT_INITIALIZER_AVAILABLE)
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation func: %s' % settings.ACTIVATION_FUNC_AVAILABLE)
parser.add_argument('--use_batch_norm',
    action='store_true', help='If set, then use batch normalization')
# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Space delimited list to change learning rate')
# Data sampler
parser.add_argument('--positive_class_sample_rates',
    nargs='+', type=float, default=[0.95], help='Space delimited list of positive class sampling rates')
parser.add_argument('--positive_class_sample_schedule',
    nargs='+', type=int, default=[-1], help='Space delimited list to change positive class sampling rate')
parser.add_argument('--positive_class_size_thresholds',
    nargs='+', type=int, default=[0], help='Space delimited list to determine positive class based on number of pixels')
# Data augmentation
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[0.50], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change learning rate')
parser.add_argument('--augmentation_rotate',
    type=float, default=-1.0,
    help='Maximum angle to rotate the image i.e. angle ~ U([-augmentation_rotate, augmentation_rotate]). \
        If less than 0, no rotation is applied.')
parser.add_argument('--augmentation_crop_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to crop')
# Segmentation loss function
parser.add_argument('--loss_func_segmentation',
    nargs='+', type=str, default=settings.LOSS_FUNC_SEGMENTATION, help='Space delimited list of loss functions: %s' %
    settings.LOSS_FUNC_AVAILABLE_SEGMENTATION)
parser.add_argument('--w_cross_entropy',
    type=float, default=settings.W_CROSS_ENTROPY, help='Weight of cross entropy loss')
parser.add_argument('--w_weight_decay_segmentation',
    type=float, default=settings.W_WEIGHT_DECAY, help='Weight of weight decay regularizer')
parser.add_argument('--w_positive_class',
    nargs='+', type=float, default=[settings.W_POSITIVE_CLASS], help='Weight of positive class penalty')
# Checkpoint settings
parser.add_argument('--n_summary',
    type=int, default=settings.N_SUMMARY, help='Number of iterations for logging summary')
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of iterations for each checkpoint')
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--segmentation_model_restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore segmentation model to resume training')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    args.device = args.device.lower()

    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    if args.segmentation_model_restore_path == '':
        args.segmentation_model_restore_path = None

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    # Check valid paths for data
    for path in args.train_multimodal_scan_paths:
        assert os.path.isfile(path)

    assert os.path.isfile(args.train_ground_truth_path)

    for path in args.val_multimodal_scan_paths:
        assert os.path.isfile(path)

    assert os.path.isfile(args.val_ground_truth_path)

    train(
        train_multimodal_scan_paths=args.train_multimodal_scan_paths,
        train_ground_truth_path=args.train_ground_truth_path,
        val_multimodal_scan_paths=args.val_multimodal_scan_paths,
        val_ground_truth_path=args.val_ground_truth_path,
        # Batch settings
        n_batch=args.n_batch,
        n_chunk=args.n_chunk,
        n_height=args.n_height,
        n_width=args.n_width,
        # Normalization settings
        dataset_normalization=args.dataset_normalization,
        dataset_means=args.dataset_means,
        dataset_stddevs=args.dataset_stddevs,
        # Segmentation network settings
        encoder_type_segmentation=args.encoder_type_segmentation,
        n_filters_encoder_segmentation=args.n_filters_encoder_segmentation,
        decoder_type_segmentation=args.decoder_type_segmentation,
        n_filters_decoder_segmentation=args.n_filters_decoder_segmentation,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        use_batch_norm=args.use_batch_norm,
        # Training settings
        learning_rates=args.learning_rates,
        learning_schedule=args.learning_schedule,
        # Data sampler
        positive_class_sample_rates=args.positive_class_sample_rates,
        positive_class_sample_schedule=args.positive_class_sample_schedule,
        positive_class_size_thresholds=args.positive_class_size_thresholds,
        # Data augmentation
        augmentation_probabilities=args.augmentation_probabilities,
        augmentation_schedule=args.augmentation_schedule,
        augmentation_rotate=args.augmentation_rotate,
        augmentation_crop_and_pad=args.augmentation_crop_and_pad,
        # Segmentation loss function
        loss_func_segmentation=args.loss_func_segmentation,
        w_cross_entropy=args.w_cross_entropy,
        w_weight_decay_segmentation=args.w_weight_decay_segmentation,
        w_positive_class=args.w_positive_class,
        # Checkpoint settings
        n_summary=args.n_summary,
        n_checkpoint=args.n_checkpoint,
        checkpoint_path=args.checkpoint_path,
        restore_path=args.segmentation_model_restore_path,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)
