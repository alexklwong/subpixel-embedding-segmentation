import argparse
import os
import global_constants as settings
from spin_main import train


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
# Subpixel embedding network settings
parser.add_argument('--encoder_type_subpixel_embedding',
    nargs='+', type=str, default=settings.ENCODER_TYPE_SUBPIXEL_EMBEDDING, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE_SUBPIXEL_EMBEDDING)
parser.add_argument('--n_filters_encoder_subpixel_embedding',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING, help='Number of filters to use in each encoder block')
parser.add_argument('--decoder_type_subpixel_embedding',
    nargs='+', type=str, default=settings.DECODER_TYPE_SUBPIXEL_EMBEDDING, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE_SUBPIXEL_EMBEDDING)
parser.add_argument('--n_filter_decoder_subpixel_embedding',
    type=int, default=settings.N_FILTER_DECODER_SUBPIXEL_EMBEDDING, help='Number of filters to use in each decoder block')
parser.add_argument('--output_channels_subpixel_embedding',
    type=int, default=settings.OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING, help='Number of filters to use in output')
parser.add_argument('--output_func_subpixel_embedding',
    type=str, default=settings.OUTPUT_FUNC, help='Output func: %s' % settings.ACTIVATION_FUNC_AVAILABLE)
# Segmentation network settings
parser.add_argument('--encoder_type_segmentation',
    nargs='+', type=str, default=settings.ENCODER_TYPE_SEGMENTATION, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE_SEGMENTATION)
parser.add_argument('--n_filters_encoder_segmentation',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_SEGMENTATION, help='Number of filters to use in each encoder block')
parser.add_argument('--resolutions_subpixel_guidance',
    nargs='+', type=int, default=settings.RESOLUTIONS_SUBPIXEL_GUIDANCE, help='Exponent of scales for subpixel guidance modules')
parser.add_argument('--n_filters_subpixel_guidance',
    nargs='+', type=int, default=settings.N_FILTERS_SUBPIXEL_GUIDANCE, help='Number of filters for each module of SPG')
parser.add_argument('--n_convolutions_subpixel_guidance',
    nargs='+', type=int, default=settings.N_CONVOLUTIONS_SUBPIXEL_GUIDANCE, help='Number of convolutions for each S2D output to undergo')
parser.add_argument('--decoder_type_segmentation',
    nargs='+', type=str, default=settings.DECODER_TYPE_SEGMENTATION, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE_SEGMENTATION)
parser.add_argument('--n_filters_decoder_segmentation',
    nargs='+', type=int, default=settings.N_FILTERS_DECODER_SEGMENTATION, help='Number of filters to use in each decoder block')
parser.add_argument('--n_filters_learnable_downsampler',
    nargs='+', type=int, default=settings.N_FILTERS_LEARNABLE_DOWNSAMPLER, help='Number of filters to use in learnable downsampler')
parser.add_argument('--kernel_sizes_learnable_downsampler',
    nargs='+', type=int, default=settings.KERNEL_SIZES_LEARNABLE_DOWNSAMPLER, help='Number of filters to use in learnable downsampler')
# Weights settings
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
parser.add_argument('--augmentation_flip_type',
    nargs='+', type=str, default=['horizontal', 'vertical'], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_rotate',
    type=float, default=-1.0,
    help='Maximum angle to rotate the image i.e. angle ~ U([-augmentation_rotate, augmentation_rotate]). \
        If less than 0, no rotation is applied.')
parser.add_argument('--augmentation_noise_type',
    type=str, default='none', help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')
parser.add_argument('--augmentation_resize_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize')
# Subpixel embedding loss function
parser.add_argument('--w_weight_decay_subpixel_embedding',
    type=float, default=settings.W_WEIGHT_DECAY, help='Weight of weight decay regularizer')
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
parser.add_argument('--restore_path',
    type=str, default=None, help='Path to restore model to resume training')
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

    if len(args.resolutions_subpixel_guidance) > 0:

        if -1 in args.n_filters_subpixel_guidance:
            args.n_filters_subpixel_guidance = [0] * len(args.resolutions_subpixel_guidance)

        if -1 in args.n_convolutions_subpixel_guidance:
            args.n_convolutions_subpixel_guidance = [0] * len(args.resolutions_subpixel_guidance)
    else:
        args.n_filters_subpixel_guidance = []
        args.n_convolutions_subpixel_guidance = []

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
        # Subpixel embedding network settings
        encoder_type_subpixel_embedding=args.encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=args.n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=args.decoder_type_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=args.n_filter_decoder_subpixel_embedding,
        output_channels_subpixel_embedding=args.output_channels_subpixel_embedding,
        output_func_subpixel_embedding=args.output_func_subpixel_embedding,
        # Segmentation network settings
        encoder_type_segmentation=args.encoder_type_segmentation,
        n_filters_encoder_segmentation=args.n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=args.resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=args.n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=args.n_convolutions_subpixel_guidance,
        decoder_type_segmentation=args.decoder_type_segmentation,
        n_filters_decoder_segmentation=args.n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=args.n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=args.kernel_sizes_learnable_downsampler,
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
        augmentation_flip_type=args.augmentation_flip_type,
        augmentation_rotate=args.augmentation_rotate,
        augmentation_noise_type=args.augmentation_noise_type,
        augmentation_noise_spread=args.augmentation_noise_spread,
        augmentation_resize_and_pad=args.augmentation_resize_and_pad,
        # Subpixel embedding loss function
        w_weight_decay_subpixel_embedding=args.w_weight_decay_subpixel_embedding,
        # Segmentation loss function
        loss_func_segmentation=args.loss_func_segmentation,
        w_cross_entropy=args.w_cross_entropy,
        w_weight_decay_segmentation=args.w_weight_decay_segmentation,
        w_positive_class=args.w_positive_class,
        # Checkpoint settings
        n_summary=args.n_summary,
        n_checkpoint=args.n_checkpoint,
        checkpoint_path=args.checkpoint_path,
        restore_path=args.restore_path,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)
