import argparse
import global_constants as settings
from spin_main import run

parser = argparse.ArgumentParser()

# validation input filepaths
parser.add_argument('--multimodal_scan_paths',
    nargs='+', type=str, required=True, help='Paths to list of MRI scan paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground-truth annotation paths')
parser.add_argument('--small_lesion_idxs_path',
    type=str, default=None, help='Pass in path to small lesion index mapping.')
# Input settings
parser.add_argument('--n_chunk',
    type=int, default=settings.N_CHUNK, help='Number of chunks or channels to process at a time')
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
# Test time augmentations
parser.add_argument('--augmentation_flip_type',
    nargs='+', type=str, default=['none'], help='Flip type for augmentation: horizontal_test, vertical_test, both_test')
# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default='', help='Path to save checkpoints')
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore segmentation model to resume training')
parser.add_argument('--do_visualize_predictions',
    action='store_true', help='If true, visualize and store predictions as png.')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')

args = parser.parse_args()

if __name__ == '__main__':

    if args.ground_truth_path == '':
        args.ground_truth_path = None

    if args.restore_path == '':
        args.restore_path = None

    if args.small_lesion_idxs_path == '':
        args.small_lesion_idxs_path = None

    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    run(multimodal_scan_paths=args.multimodal_scan_paths,
        ground_truth_path=args.ground_truth_path,
        small_lesion_idxs_path=args.small_lesion_idxs_path,
        # Input settings
        n_chunk=args.n_chunk,
        # Normalization setting
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
        # Weights settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        use_batch_norm=args.use_batch_norm,
        # Test time augmentation
        augmentation_flip_type=args.augmentation_flip_type,
        # Checkpoint settings
        checkpoint_path=args.checkpoint_path,
        restore_path=args.restore_path,
        do_visualize_predictions=args.do_visualize_predictions,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)
