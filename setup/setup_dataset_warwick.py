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

# Paths
WARWICK_RAW_DATA_DIRPATH = os.path.join('data', 'warwick')
WARWICK_NUMPY_DATA_DIRPATH = os.path.join('data', 'warwick_spin')

# Paths to the text files to hold all paths to numpy data of scans and ground truth annotations
SCAN_OUTPUT_FILEPATH = os.path.join(WARWICK_NUMPY_DATA_DIRPATH, 'scans.txt')
GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(WARWICK_NUMPY_DATA_DIRPATH, 'ground_truths.txt')

# Paths to training, validation, and testing splits: train-test
DATA_SPLIT_DIRPATH = os.path.join('data_split', 'warwick')

TRAIN_REF_DIRPATH = os.path.join('training', 'warwick')
TEST_REF_DIRPATH = os.path.join('testing', 'warwick')

# Train test split output paths
TRAIN_TRAINTEST_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'traintest')
TEST_TRAINTEST_REF_DIRPATH = os.path.join(TEST_REF_DIRPATH, 'traintest')

TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'warwick_train_scans.txt')
TRAIN_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'warwick_train_ground_truths.txt')
TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'warwick_test_scans.txt')
TEST_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'warwick_test_ground_truths.txt')

# Lists to hold the paths to numpy arrays. Eventually written out to text files
scan_output_paths = []
annotated_output_paths = []

train_traintest_scan_input_paths = []
train_traintest_ground_truth_input_paths = []
test_traintest_scan_input_paths = []
test_traintest_ground_truth_input_paths = []

train_traintest_scan_output_paths = []
train_traintest_ground_truth_output_paths = []
test_traintest_scan_output_paths = []
test_traintest_ground_truth_output_paths = []

OUTPUT_DIRECTORIES = [
    TRAIN_TRAINTEST_REF_DIRPATH,
    TEST_TRAINTEST_REF_DIRPATH,
    WARWICK_NUMPY_DATA_DIRPATH
]

def bmp_to_numpy(scan_path):
    '''
    Convert the .bmp file given to a numpy array

    Arg(s):
        scan_path : str
            path from the current working directory to the .bmp file
    Returns:
        np.array : scan
    '''

    return np.asarray(Image.open(scan_path)).astype('float32')

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
        numpy_file = bmp_to_numpy(raw_filepath)
        np.save(numpy_filepath, numpy_file)

def store_scan_paths():
    '''
    Stores paths to .npy data and converts data if necessary

    Arg(s):
        None

    Returns:
        None
    '''
    if not os.path.exists(WARWICK_RAW_DATA_DIRPATH):
        print("Path to raw data {} does not exist.".format(WARWICK_RAW_DATA_DIRPATH))
        return

    for filename in os.listdir(WARWICK_RAW_DATA_DIRPATH):
        _, extension = os.path.splitext(filename)
        numpy_filename = filename.replace(extension, '.npy')
        numpy_save_path = os.path.join(WARWICK_NUMPY_DATA_DIRPATH, numpy_filename)
        raw_save_path = os.path.join(WARWICK_RAW_DATA_DIRPATH, filename)
        if not extension == '.bmp':
            continue
        if 'train' in filename:
            if 'anno' in filename:
                train_traintest_ground_truth_input_paths.append(raw_save_path)
                train_traintest_ground_truth_output_paths.append(numpy_save_path)
            else:
                train_traintest_scan_input_paths.append(raw_save_path)
                train_traintest_scan_output_paths.append(numpy_save_path)
        elif 'test' in filename:
            if 'anno' in filename:
                test_traintest_ground_truth_input_paths.append(raw_save_path)
                test_traintest_ground_truth_output_paths.append(numpy_save_path)
            else:
                test_traintest_scan_input_paths.append(raw_save_path)
                test_traintest_scan_output_paths.append(numpy_save_path)
 
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
    assert len(train_traintest_ground_truth_input_paths) == len(train_traintest_ground_truth_output_paths)
    assert len(test_traintest_scan_input_paths) == len(test_traintest_scan_output_paths)
    assert len(test_traintest_ground_truth_input_paths) == len(test_traintest_ground_truth_output_paths)

    raw_paths = train_traintest_scan_input_paths + \
        train_traintest_ground_truth_input_paths + \
        test_traintest_scan_input_paths + \
        test_traintest_ground_truth_input_paths

    numpy_paths = train_traintest_scan_output_paths + \
        train_traintest_ground_truth_output_paths + \
        test_traintest_scan_output_paths + \
        test_traintest_ground_truth_output_paths

    # Convert from .bmp -> .npy
    if not paths_only:
        convert_all_to_numpy(raw_paths, numpy_paths)
    
    # Write .txt files
    data_utils.write_paths(TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH, sorted(train_traintest_scan_output_paths))
    data_utils.write_paths(TRAIN_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH, sorted(train_traintest_ground_truth_output_paths))
    data_utils.write_paths(TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH, sorted(test_traintest_scan_output_paths))
    data_utils.write_paths(TEST_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH, sorted(test_traintest_ground_truth_output_paths))

if __name__ == '__main__':
    setup_utils.create_output_directories(OUTPUT_DIRECTORIES)

    setup_dataset(args.paths_only)

    # Status print statements
    if not args.paths_only:
        print("Saved .npy files in {}".format(WARWICK_NUMPY_DATA_DIRPATH))
    print("Saved training scan paths in {}".format(TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH))
    print("Saved training ground truth paths in {}".format(TRAIN_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH))
    print("Saved test scan paths in {}".format(TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH))
    print("Saved test ground truth paths in {}".format(TEST_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH))

