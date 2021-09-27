# Batch settings
N_BATCH                                 = 8
O_HEIGHT                                = 197
O_WIDTH                                 = 233
N_CHUNK                                 = 1
N_HEIGHT                                = 192
N_WIDTH                                 = 256
N_CHANNEL                               = 189

NORMALIZATION_TYPE_AVAILABLE            = ['none', 'min-max', 'mean-sd']

EPS                                     = 1e-6
# Segmentation network settings
ENCODER_TYPE_AVAILABLE_SEGMENTATION     = ['resnet18',
                                           'vggnet13']
DECODER_TYPE_AVAILABLE_SEGMENTATION     = ['generic']
DECONV_TYPE_AVAILABLE_SEGMENTATION      = ['transpose', 'up']
ENCODER_TYPE_SEGMENTATION               = 'vggnet13'
INPUT_CHANNELS_SEGMENTATION             = 1
N_FILTERS_ENCODER_SEGMENTATION          = [64, 128, 256, 512, 1024]
DECODER_TYPE_SEGMENTATION               = 'generic'
N_FILTERS_DECODER_SEGMENTATION          = [512, 256, 128, 64]
DECONV_TYPE_SEGMENTATION                = 'up'

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
LEARNING_RATES                          = [3.00e-4, 1.00e-4, 5.0e-5]
LEARNING_SCHEDULE                       = [400, 1400, 1600]
USE_AUGMENT                             = False
FLIP_TYPE                               = ['none']
POSITIVE_CLASS_SAMPLE_RATE              = 0.90
LOSS_FUNC_AVAILABLE                     = ['cross_entropy', 'weight_decay']
LOSS_FUNC                               = ['cross_entropy']
W_CROSS_ENTROPY                         = 1.00
W_DICE                                  = 1.00
W_WEIGHT_DECAY                          = 1e-4

W_POSITIVE_CLASS                        = 1.00

LOSS_FUNC_AVAILABLE_SEGMENTATION        = ['cross_entropy',
                                           'dice',
                                           'weight_decay']
LOSS_FUNC_SEGMENTATION                  = ['cross_entropy',
                                           'weight_decay']

# Checkpoint settings
N_SUMMARY                               = 1000
N_CHECKPOINT                            = 5000
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
