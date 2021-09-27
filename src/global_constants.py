# Batch settings
N_BATCH                                 = 8
O_HEIGHT                                = 197
O_WIDTH                                 = 233
N_CHUNK                                 = 1
N_HEIGHT                                = 192
N_WIDTH                                 = 256
N_CHANNEL                               = 189

EPS                                     = 1e-6

# Segmentation network settings
ENCODER_TYPE_AVAILABLE_SEGMENTATION     = ['resnet18',
                                           'resnet34',
                                           'resnet50']
DECODER_TYPE_AVAILABLE_SEGMENTATION     = ['subpixel_guidance',
                                           'learnable_downsampler',
                                           'generic']
ENCODER_TYPE_SEGMENTATION               = 'resnet18'
INPUT_CHANNELS_SEGMENTATION             = 1
N_FILTERS_ENCODER_SEGMENTATION          = [32, 64, 128, 196, 196]
DECODER_TYPE_SEGMENTATION               = 'generic'
N_FILTERS_DECODER_SEGMENTATION          = [196, 128, 64, 32, 16]

# Subpixel Embedding network settings
ENCODER_TYPE_AVAILABLE_SUBPIXEL_EMBEDDING = ['resnet5_super_resolution',
                                             'resnet7_super_resolution',
                                             'none']
DECODER_TYPE_AVAILABLE_SUBPIXEL_EMBEDDING = ['subpixel', 'none']
ENCODER_TYPE_SUBPIXEL_EMBEDDING           = 'resnet5_super_resolution'
N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING      = [16, 16, 16]
DECODER_TYPE_SUBPIXEL_EMBEDDING           = 'sub_pixel'
N_FILTER_DECODER_SUBPIXEL_EMBEDDING       = 16
OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING        = 8

# Subpixel Guidance network settings
RESOLUTIONS_SUBPIXEL_GUIDANCE           = [0, 1]
N_FILTERS_SUBPIXEL_GUIDANCE             = [8, 8]
N_CONVOLUTIONS_SUBPIXEL_GUIDANCE        = [1, 1]

# Learnable Downsampler network settings
N_FILTERS_LEARNABLE_DOWNSAMPLER         = [16, 16]
KERNEL_SIZES_LEARNABLE_DOWNSAMPLER      = [3, 3]

# General weight settings
WEIGHT_INITIALIZER_AVAILABLE            = ['kaiming_normal',
                                           'kaiming_uniform',
                                           'xavier_normal',
                                           'xavier_uniform']
ACTIVATION_FUNC_AVAILABLE               = ['relu', 'leaky_relu', 'elu']
WEIGHT_INITIALIZER                      = 'kaiming_uniform'
ACTIVATION_FUNC                         = 'leaky_relu'
OUTPUT_FUNC                             = 'linear'
USE_BATCH_NORM                          = False

# Training settings
N_EPOCH                                 = 1600
LEARNING_RATES                          = [3.00e-4, 1.00e-4, 5.00e-5]
LEARNING_SCHEDULE                       = [400, 1400, 1600]
W_CROSS_ENTROPY                         = 1.00
W_WEIGHT_DECAY                          = 1e-4
W_POSITIVE_CLASS                        = 1.00

LOSS_FUNC_AVAILABLE_SEGMENTATION        = ['cross_entropy',
                                           'dice',
                                           'weight_decay']
LOSS_FUNC_SEGMENTATION                  = ['cross_entropy',
                                           'weight_decay']

# Checkpoint settings
N_DISPLAY                               = 4
N_SUMMARY                               = 1000
N_CHECKPOINT                            = 500
CHECKPOINT_PATH                         = ''
RESTORE_PATH                            = ''

# Hardware settings
DEVICE                                  = 'cuda'
CUDA                                    = 'cuda'
CPU                                     = 'cpu'
GPU                                     = 'gpu'
N_THREAD                                = 8

# Dataset Constants
ATLAS_MIN                               = -3.316535e-05
ATLAS_MAX                               = 100.00001
ATLAS_MEAN                              = 30.20063
ATLAS_SD                                = 35.221165