import os, sys
import numpy as np
import argparse
from PIL import Image
sys.path.insert(0, 'src')
import setup_utils
import data_utils

DEBUG = False
parser = argparse.ArgumentParser()
parser.add_argument('--paths_only', action='store_true')
args = parser.parse_args()

# Raw data paths
RITE_RAW_DATA_DIRPATH = os.path.join('data', 'AV_groundTruth')
RITE_RAW_TEST_DATA_DIRPATH = os.path.join(RITE_RAW_DATA_DIRPATH, 'test')
RITE_RAW_TRAIN_DATA_DIRPATH = os.path.join(RITE_RAW_DATA_DIRPATH, 'training')
# Test
RITE_RAW_TEST_SCAN_DIRPATH = os.path.join(RITE_RAW_TEST_DATA_DIRPATH, 'images')
RITE_RAW_TEST_GROUND_TRUTHS_VESSEL_DIRPATH = os.path.join(RITE_RAW_TEST_DATA_DIRPATH, 'vessel')
RITE_RAW_TEST_GROUND_TRUTHS_AV_DIRPATH = os.path.join(RITE_RAW_TEST_DATA_DIRPATH, 'av')
# Train
RITE_RAW_TRAIN_SCAN_DIRPATH = os.path.join(RITE_RAW_TRAIN_DATA_DIRPATH, 'images')
RITE_RAW_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH = os.path.join(RITE_RAW_TRAIN_DATA_DIRPATH, 'vessel')
RITE_RAW_TRAIN_GROUND_TRUTHS_AV_DIRPATH = os.path.join(RITE_RAW_TRAIN_DATA_DIRPATH, 'av')

# Numpy data paths
RITE_NUMPY_DATA_DIRPATH = os.path.join('data', 'rite_spin')
RITE_NUMPY_TEST_DATA_DIRPATH = os.path.join(RITE_NUMPY_DATA_DIRPATH, 'test')
RITE_NUMPY_TRAIN_DATA_DIRPATH = os.path.join(RITE_NUMPY_DATA_DIRPATH, 'training')
# Test
RITE_NUMPY_TEST_SCAN_DIRPATH = os.path.join(RITE_NUMPY_TEST_DATA_DIRPATH, 'images')
RITE_NUMPY_TEST_GROUND_TRUTHS_VESSEL_DIRPATH = os.path.join(RITE_NUMPY_TEST_DATA_DIRPATH, 'vessel')
RITE_NUMPY_TEST_GROUND_TRUTHS_AV_DIRPATH = os.path.join(RITE_NUMPY_TEST_DATA_DIRPATH, 'av')
# Train
RITE_NUMPY_TRAIN_SCAN_DIRPATH = os.path.join(RITE_NUMPY_TRAIN_DATA_DIRPATH, 'images')
RITE_NUMPY_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH = os.path.join(RITE_NUMPY_TRAIN_DATA_DIRPATH, 'vessel')
RITE_NUMPY_TRAIN_GROUND_TRUTHS_AV_DIRPATH = os.path.join(RITE_NUMPY_TRAIN_DATA_DIRPATH, 'av')

MAP = [
    ['']
]
# Paths to the text files to hold all paths to numpy data of scans and ground truth annotations
# SCAN_OUTPUT_FILEPATH = os.path.join(RITE_NUMPY_DATA_DIRPATH, 'scans.txt')
# GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(RITE_NUMPY_DATA_DIRPATH, 'ground_truths.txt')

# Paths to training, validation, and testing splits: train-test
DATA_SPLIT_DIRPATH = os.path.join('data_split', 'rite')

TRAIN_REF_DIRPATH = os.path.join('training', 'rite')
TEST_REF_DIRPATH = os.path.join('testing', 'rite')

# Train test split output paths
TRAIN_TRAINTEST_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'traintest')
TEST_TRAINTEST_REF_DIRPATH = os.path.join(TEST_REF_DIRPATH, 'traintest')

TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'rite_train_scans.txt')
TRAIN_TRAINTEST_GROUND_TRUTH_VESSEL_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'rite_train_ground_truths_vessel.txt')
TRAIN_TRAINTEST_GROUND_TRUTH_AV_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'rite_train_ground_truths_av.txt')
TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'rite_test_scans.txt')
TEST_TRAINTEST_GROUND_TRUTH_VESSEL_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'rite_test_ground_truths_vessel.txt')
TEST_TRAINTEST_GROUND_TRUTH_AV_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'rite_test_ground_truths_av.txt')

# Lists to hold the paths to numpy arrays. Eventually written out to text files
scan_output_paths = []
annotated_output_paths = []

train_traintest_scan_input_paths = []
train_traintest_ground_truth_vessel_input_paths = []
train_traintest_ground_truth_av_input_paths = []
test_traintest_scan_input_paths = []
test_traintest_ground_truth_vessel_input_paths = []
test_traintest_ground_truth_av_input_paths = []

train_traintest_scan_output_paths = []
train_traintest_ground_truth_vessel_output_paths = []
train_traintest_ground_truth_av_output_paths = []
test_traintest_scan_output_paths = []
test_traintest_ground_truth_vessel_output_paths = []
test_traintest_ground_truth_av_output_paths = []

OUTPUT_DIRECTORIES = [
    TRAIN_TRAINTEST_REF_DIRPATH,
    TEST_TRAINTEST_REF_DIRPATH,
    RITE_NUMPY_TEST_SCAN_DIRPATH,
    RITE_NUMPY_TEST_GROUND_TRUTHS_VESSEL_DIRPATH,
    RITE_NUMPY_TEST_GROUND_TRUTHS_AV_DIRPATH,
    RITE_NUMPY_TRAIN_SCAN_DIRPATH,
    RITE_NUMPY_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH,
    RITE_NUMPY_TRAIN_GROUND_TRUTHS_AV_DIRPATH
]

# Map raw data dirpaths to numpy data dirpaths in a list
# MAP = {
#     'test': {
#         'raw_scan': RITE_RAW_TEST_SCAN_DIRPATH,
#     }
#     RITE_RAW_TEST_SCAN_DIRPATH: RITE_NUMPY_TEST_SCAN_DIRPATH,
#     RITE_RAW_TEST_GROUND_TRUTHS_VESSEL_DIRPATH: RITE_NUMPY_TEST_GROUND_TRUTHS_VESSEL_DIRPATH,
#     RITE_RAW_TEST_GROUND_TRUTHS_AV_DIRPATH: RITE_NUMPY_TEST_GROUND_TRUTHS_AV_DIRPATH,
#     RITE_RAW_TRAIN_SCAN_DIRPATH: RITE_NUMPY_TRAIN_SCAN_DIRPATH,
#     RITE_RAW_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH: RITE_NUMPY_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH,
#     RITE_RAW_TRAIN_GROUND_TRUTHS_AV_DIRPATH: RITE_NUMPY_TRAIN_GROUND_TRUTHS_AV_DIRPATH
# }

def to_numpy(scan_path):
    '''
    Convert the .png file given to a numpy array

    Arg(s):
        scan_path : str
            path from the current working directory to the .png file
    Returns:
        np.array : scan
    '''

    return np.array(Image.open(scan_path)).astype('float32')

def convert_all_to_numpy(input_paths, output_paths):
    '''
    Convert all the files at input_path and save to corresponding output_path

    Arg(s):
        input_paths : list[str]
            input .bmp files to be converted
        output_paths : list[str]
            destination to save .npy files

    Returns:
        None
    '''

    assert len(input_paths) == len(output_paths)
    for raw_filepath, numpy_filepath in zip(input_paths, output_paths):
        numpy_file = to_numpy(raw_filepath)
        np.save(numpy_filepath, numpy_file)

def store_scan_paths():
    '''
    Stores paths to .npy data and converts data if necessary

    Arg(s):
        None

    Returns:
        None
    '''
    if not os.path.exists(RITE_RAW_DATA_DIRPATH):
        raise ValueError("Path to raw data {} does not exist.".format(RITE_RAW_DATA_DIRPATH))

    # Test data
    for raw_scan_filename in os.listdir(RITE_RAW_TEST_SCAN_DIRPATH):
        # Raw filepaths
        raw_ground_truth_filename = raw_scan_filename.replace('tif', 'png')
        raw_scan_filepath = os.path.join(RITE_RAW_TEST_SCAN_DIRPATH, raw_scan_filename)
        raw_ground_truth_vessel_filepath = os.path.join(RITE_RAW_TEST_GROUND_TRUTHS_VESSEL_DIRPATH, raw_ground_truth_filename)
        raw_ground_truth_av_filepath = os.path.join(RITE_RAW_TEST_GROUND_TRUTHS_AV_DIRPATH, raw_ground_truth_filename)
        # print("scan: {}\nvessel: {}\nav: {}".format(raw_scan_filepath,raw_ground_truth_vessel_filepath, raw_ground_truth_av_filepath))
        # Numpy filepaths
        numpy_filename = raw_scan_filename.replace('tif', 'npy')
        numpy_scan_filepath = os.path.join(RITE_NUMPY_TEST_SCAN_DIRPATH, numpy_filename.replace('.npy', '_image.npy'))
        numpy_ground_truth_vessel_filepath = os.path.join(RITE_NUMPY_TEST_GROUND_TRUTHS_VESSEL_DIRPATH, numpy_filename.replace('.npy', '_vessel.npy'))
        numpy_ground_truth_av_filepath = os.path.join(RITE_NUMPY_TEST_GROUND_TRUTHS_AV_DIRPATH, numpy_filename.replace('.npy', '_av.npy'))

        # Store scan paths (raw and numpy)
        test_traintest_scan_input_paths.append(raw_scan_filepath)
        test_traintest_scan_output_paths.append(numpy_scan_filepath)
        # Store vessel paths (raw and numpy)
        test_traintest_ground_truth_vessel_input_paths.append(raw_ground_truth_vessel_filepath)
        test_traintest_ground_truth_vessel_output_paths.append(numpy_ground_truth_vessel_filepath)
        # Store av paths (raw and numpy)
        test_traintest_ground_truth_av_input_paths.append(raw_ground_truth_av_filepath)
        test_traintest_ground_truth_av_output_paths.append(numpy_ground_truth_av_filepath)

    # Train data
    for raw_scan_filename in os.listdir(RITE_RAW_TRAIN_SCAN_DIRPATH):
        # Raw filepaths
        raw_ground_truth_filename = raw_scan_filename.replace('tif', 'png')
        raw_scan_filepath = os.path.join(RITE_RAW_TRAIN_SCAN_DIRPATH, raw_scan_filename)
        raw_ground_truth_vessel_filepath = os.path.join(RITE_RAW_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH, raw_ground_truth_filename)
        raw_ground_truth_av_filepath = os.path.join(RITE_RAW_TRAIN_GROUND_TRUTHS_AV_DIRPATH, raw_ground_truth_filename)
        # Numpy filepaths
        numpy_filename = raw_scan_filename.replace('tif', 'npy')
        numpy_scan_filepath = os.path.join(RITE_NUMPY_TRAIN_SCAN_DIRPATH, numpy_filename.replace('.npy', '_image.npy'))
        numpy_ground_truth_vessel_filepath = os.path.join(RITE_NUMPY_TRAIN_GROUND_TRUTHS_VESSEL_DIRPATH, numpy_filename.replace('.npy', '_vessel.npy'))
        numpy_ground_truth_av_filepath = os.path.join(RITE_NUMPY_TRAIN_GROUND_TRUTHS_AV_DIRPATH, numpy_filename.replace('.npy', '_av.npy'))

        # Store scan paths (raw and numpy)
        train_traintest_scan_input_paths.append(raw_scan_filepath)
        train_traintest_scan_output_paths.append(numpy_scan_filepath)
        # Store vessel paths (raw and numpy)
        train_traintest_ground_truth_vessel_input_paths.append(raw_ground_truth_vessel_filepath)
        train_traintest_ground_truth_vessel_output_paths.append(numpy_ground_truth_vessel_filepath)
        # Store av paths (raw and numpy)
        train_traintest_ground_truth_av_input_paths.append(raw_ground_truth_av_filepath)
        train_traintest_ground_truth_av_output_paths.append(numpy_ground_truth_av_filepath)

def setup_dataset(paths_only=False):
    '''
    Writes paths to train and test data to .txt files & converts data to numpy if necessary

    Arg(s):
        paths_only : bool
            if true, do not convert data to .npy and only save paths

    Returns:
        None
    '''
    # Store paths in global lists
    store_scan_paths()

    assert len(train_traintest_scan_input_paths) == len(train_traintest_scan_output_paths)
    assert len(train_traintest_ground_truth_av_input_paths) == len(train_traintest_ground_truth_av_output_paths)
    assert len(train_traintest_ground_truth_vessel_input_paths) == len(train_traintest_ground_truth_vessel_output_paths)
    assert len(test_traintest_scan_input_paths) == len(test_traintest_scan_output_paths)
    assert len(test_traintest_ground_truth_av_input_paths) == len(test_traintest_ground_truth_av_output_paths)
    assert len(test_traintest_ground_truth_vessel_input_paths) == len(test_traintest_ground_truth_vessel_output_paths)

    # Convert from .bmp -> .npy
    if not paths_only:
        raw_paths = train_traintest_scan_input_paths + \
            train_traintest_ground_truth_av_input_paths + \
            train_traintest_ground_truth_vessel_input_paths + \
            test_traintest_scan_input_paths + \
            test_traintest_ground_truth_av_input_paths + \
            test_traintest_ground_truth_vessel_input_paths

        numpy_paths = train_traintest_scan_output_paths + \
            train_traintest_ground_truth_av_output_paths + \
            train_traintest_ground_truth_vessel_output_paths + \
            test_traintest_scan_output_paths + \
            test_traintest_ground_truth_av_output_paths + \
            test_traintest_ground_truth_vessel_output_paths

        convert_all_to_numpy(raw_paths, numpy_paths)
        print("Saved .npy files in {}".format(RITE_NUMPY_DATA_DIRPATH))

    # Sort lists
    train_traintest_scan_output_paths.sort()
    train_traintest_ground_truth_av_output_paths.sort()
    train_traintest_ground_truth_vessel_output_paths.sort()
    test_traintest_scan_output_paths.sort()
    test_traintest_ground_truth_av_output_paths.sort()
    test_traintest_ground_truth_vessel_output_paths.sort()

    # Write paths
    data_utils.write_paths(TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH, train_traintest_scan_output_paths)
    data_utils.write_paths(TRAIN_TRAINTEST_GROUND_TRUTH_AV_OUTPUT_FILEPATH, train_traintest_ground_truth_av_output_paths)
    data_utils.write_paths(TRAIN_TRAINTEST_GROUND_TRUTH_VESSEL_OUTPUT_FILEPATH, train_traintest_ground_truth_vessel_output_paths)
    data_utils.write_paths(TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH, test_traintest_scan_output_paths)
    data_utils.write_paths(TEST_TRAINTEST_GROUND_TRUTH_AV_OUTPUT_FILEPATH, test_traintest_ground_truth_av_output_paths)
    data_utils.write_paths(TEST_TRAINTEST_GROUND_TRUTH_VESSEL_OUTPUT_FILEPATH, test_traintest_ground_truth_vessel_output_paths)

    # Print progress
    print("Saved training scan paths in {}".format(TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH))
    print("Saved training ground truth av paths in {}".format(TRAIN_TRAINTEST_GROUND_TRUTH_AV_OUTPUT_FILEPATH))
    print("Saved training ground truth vessel paths in {}".format(TRAIN_TRAINTEST_GROUND_TRUTH_VESSEL_OUTPUT_FILEPATH))
    print("Saved test scan paths in {}".format(TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH))
    print("Saved test ground truth av paths in {}".format(TEST_TRAINTEST_GROUND_TRUTH_AV_OUTPUT_FILEPATH))
    print("Saved test ground truth vessel paths in {}".format(TEST_TRAINTEST_GROUND_TRUTH_VESSEL_OUTPUT_FILEPATH))

if __name__ == '__main__':
    # Create output directories
    setup_utils.create_output_directories(OUTPUT_DIRECTORIES)

    setup_dataset(args.paths_only)

