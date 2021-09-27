import os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
import data_utils


DEBUG = False


# Paths
ATLAS_RAW_DATA_DIRPATH = os.path.join('data', 'atlas', 'atlas_standard')
ATLAS_NUMPY_DATA_DIRPATH = os.path.join('data', 'atlas_spin')

# Paths to the text files to hold all paths to numpy data of scans and ground truth annotations
SCAN_OUTPUT_FILEPATH = os.path.join(ATLAS_NUMPY_DATA_DIRPATH, 'scans.txt')
GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(ATLAS_NUMPY_DATA_DIRPATH, 'ground_truths.txt')

# Paths to training, validation, and testing splits: train-test, 6-fold
DATA_SPLIT_DIRPATH = os.path.join('data_split', 'atlas')

TRAIN_REF_DIRPATH = os.path.join('training', 'atlas')
VAL_REF_DIRPATH = os.path.join('validation', 'atlas')
TEST_REF_DIRPATH = os.path.join('testing', 'atlas')

# Train test split input paths
TRAIN_TRAINTEST_SCAN_INPUT_FILEPATH = \
    os.path.join(DATA_SPLIT_DIRPATH, 'traintest', 'training', 'train_scans.txt')
TRAIN_TRAINTEST_GROUND_TRUTH_INPUT_FILEPATH = \
    os.path.join(DATA_SPLIT_DIRPATH, 'traintest', 'training', 'train_ground_truths.txt')
TEST_TRAINTEST_SCAN_INPUT_FILEPATH = \
    os.path.join(DATA_SPLIT_DIRPATH, 'traintest', 'testing', 'test_scans.txt')
TEST_TRAINTEST_GROUND_TRUTH_INPUT_FILEPATH = \
    os.path.join(DATA_SPLIT_DIRPATH, 'traintest', 'testing', 'test_ground_truths.txt')

# Train test split output paths
TRAIN_TRAINTEST_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'traintest')
TEST_TRAINTEST_REF_DIRPATH = os.path.join(TEST_REF_DIRPATH, 'traintest')

TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'atlas_train_scans.txt')
TRAIN_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_TRAINTEST_REF_DIRPATH, 'atlas_train_ground_truths.txt')
TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'atlas_test_scans.txt')
TEST_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_TRAINTEST_REF_DIRPATH, 'atlas_test_ground_truths.txt')

# 6-fold split input paths
TRAIN_6FOLD_SCAN1_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'scans', 'train_scans-1.txt')
TRAIN_6FOLD_SCAN2_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'scans', 'train_scans-2.txt')
TRAIN_6FOLD_SCAN3_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'scans', 'train_scans-3.txt')
TRAIN_6FOLD_SCAN4_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'scans', 'train_scans-4.txt')
TRAIN_6FOLD_SCAN5_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'scans', 'train_scans-5.txt')
TRAIN_6FOLD_SCAN6_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'scans', 'train_scans-6.txt')
TRAIN_6FOLD_GROUND_TRUTH1_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'ground_truths', 'train_ground_truths-1.txt')
TRAIN_6FOLD_GROUND_TRUTH2_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'ground_truths', 'train_ground_truths-2.txt')
TRAIN_6FOLD_GROUND_TRUTH3_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'ground_truths', 'train_ground_truths-3.txt')
TRAIN_6FOLD_GROUND_TRUTH4_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'ground_truths', 'train_ground_truths-4.txt')
TRAIN_6FOLD_GROUND_TRUTH5_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'ground_truths', 'train_ground_truths-5.txt')
TRAIN_6FOLD_GROUND_TRUTH6_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'training', 'ground_truths', 'train_ground_truths-6.txt')
VAL_6FOLD_SCAN1_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'scans', 'val_scans-1.txt')
VAL_6FOLD_SCAN2_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'scans', 'val_scans-2.txt')
VAL_6FOLD_SCAN3_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'scans', 'val_scans-3.txt')
VAL_6FOLD_SCAN4_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'scans', 'val_scans-4.txt')
VAL_6FOLD_SCAN5_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'scans', 'val_scans-5.txt')
VAL_6FOLD_SCAN6_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'scans', 'val_scans-6.txt')
VAL_6FOLD_GROUND_TRUTH1_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'ground_truths', 'val_ground_truths-1.txt')
VAL_6FOLD_GROUND_TRUTH2_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'ground_truths', 'val_ground_truths-2.txt')
VAL_6FOLD_GROUND_TRUTH3_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'ground_truths', 'val_ground_truths-3.txt')
VAL_6FOLD_GROUND_TRUTH4_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'ground_truths', 'val_ground_truths-4.txt')
VAL_6FOLD_GROUND_TRUTH5_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'ground_truths', 'val_ground_truths-5.txt')
VAL_6FOLD_GROUND_TRUTH6_INPUT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, '6fold', 'validation', 'ground_truths', 'val_ground_truths-6.txt')

# 6-fold split output paths
TRAIN_6FOLD_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, '6fold')
TRAIN_6FOLD_SCAN_REF_DIRPATH = os.path.join(TRAIN_6FOLD_REF_DIRPATH, 'scans')
TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH = os.path.join(TRAIN_6FOLD_REF_DIRPATH, 'ground_truths')
VAL_6FOLD_REF_DIRPATH = os.path.join(VAL_REF_DIRPATH, '6fold')
VAL_6FOLD_SCAN_REF_DIRPATH = os.path.join(VAL_6FOLD_REF_DIRPATH, 'scans')
VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH = os.path.join(VAL_6FOLD_REF_DIRPATH, 'ground_truths')

TRAIN_6FOLD_SCAN1_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_SCAN_REF_DIRPATH, 'atlas_train_scans-1.txt')
TRAIN_6FOLD_SCAN2_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_SCAN_REF_DIRPATH, 'atlas_train_scans-2.txt')
TRAIN_6FOLD_SCAN3_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_SCAN_REF_DIRPATH, 'atlas_train_scans-3.txt')
TRAIN_6FOLD_SCAN4_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_SCAN_REF_DIRPATH, 'atlas_train_scans-4.txt')
TRAIN_6FOLD_SCAN5_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_SCAN_REF_DIRPATH, 'atlas_train_scans-5.txt')
TRAIN_6FOLD_SCAN6_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_SCAN_REF_DIRPATH, 'atlas_train_scans-6.txt')
TRAIN_6FOLD_GROUND_TRUTH1_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_train_ground_truths-1.txt')
TRAIN_6FOLD_GROUND_TRUTH2_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_train_ground_truths-2.txt')
TRAIN_6FOLD_GROUND_TRUTH3_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_train_ground_truths-3.txt')
TRAIN_6FOLD_GROUND_TRUTH4_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_train_ground_truths-4.txt')
TRAIN_6FOLD_GROUND_TRUTH5_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_train_ground_truths-5.txt')
TRAIN_6FOLD_GROUND_TRUTH6_OUTPUT_FILEPATH = os.path.join(TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_train_ground_truths-6.txt')
VAL_6FOLD_SCAN1_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_SCAN_REF_DIRPATH, 'atlas_val_scans-1.txt')
VAL_6FOLD_SCAN2_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_SCAN_REF_DIRPATH, 'atlas_val_scans-2.txt')
VAL_6FOLD_SCAN3_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_SCAN_REF_DIRPATH, 'atlas_val_scans-3.txt')
VAL_6FOLD_SCAN4_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_SCAN_REF_DIRPATH, 'atlas_val_scans-4.txt')
VAL_6FOLD_SCAN5_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_SCAN_REF_DIRPATH, 'atlas_val_scans-5.txt')
VAL_6FOLD_SCAN6_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_SCAN_REF_DIRPATH, 'atlas_val_scans-6.txt')
VAL_6FOLD_GROUND_TRUTH1_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_val_ground_truths-1.txt')
VAL_6FOLD_GROUND_TRUTH2_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_val_ground_truths-2.txt')
VAL_6FOLD_GROUND_TRUTH3_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_val_ground_truths-3.txt')
VAL_6FOLD_GROUND_TRUTH4_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_val_ground_truths-4.txt')
VAL_6FOLD_GROUND_TRUTH5_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_val_ground_truths-5.txt')
VAL_6FOLD_GROUND_TRUTH6_OUTPUT_FILEPATH = os.path.join(VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH, 'atlas_val_ground_truths-6.txt')

# Lists to hold the paths to numpy arrays. Eventually written out to text files
scan_output_paths = []
annotated_output_paths = []

# Lists to hold paths to train val
train_trainval_scan_input_paths = data_utils.read_paths(TRAIN_TRAINTEST_SCAN_INPUT_FILEPATH)
train_trainval_ground_truth_input_paths = data_utils.read_paths(TRAIN_TRAINTEST_GROUND_TRUTH_INPUT_FILEPATH)
val_trainval_scan_input_paths = data_utils.read_paths(TEST_TRAINTEST_SCAN_INPUT_FILEPATH)
val_trainval_ground_truth_input_paths = data_utils.read_paths(TEST_TRAINTEST_GROUND_TRUTH_INPUT_FILEPATH)

train_trainval_scan_output_paths = []
train_trainval_ground_truth_output_paths = []
val_trainval_scan_output_paths = []
val_trainval_ground_truth_output_paths = []

# Lists to hold paths to 6-fold
train_6fold_scan1_input_paths = data_utils.read_paths(TRAIN_6FOLD_SCAN1_INPUT_FILEPATH)
train_6fold_scan2_input_paths = data_utils.read_paths(TRAIN_6FOLD_SCAN2_INPUT_FILEPATH)
train_6fold_scan3_input_paths = data_utils.read_paths(TRAIN_6FOLD_SCAN3_INPUT_FILEPATH)
train_6fold_scan4_input_paths = data_utils.read_paths(TRAIN_6FOLD_SCAN4_INPUT_FILEPATH)
train_6fold_scan5_input_paths = data_utils.read_paths(TRAIN_6FOLD_SCAN5_INPUT_FILEPATH)
train_6fold_scan6_input_paths = data_utils.read_paths(TRAIN_6FOLD_SCAN6_INPUT_FILEPATH)
train_6fold_ground_truth1_input_paths = data_utils.read_paths(TRAIN_6FOLD_GROUND_TRUTH1_INPUT_FILEPATH)
train_6fold_ground_truth2_input_paths = data_utils.read_paths(TRAIN_6FOLD_GROUND_TRUTH2_INPUT_FILEPATH)
train_6fold_ground_truth3_input_paths = data_utils.read_paths(TRAIN_6FOLD_GROUND_TRUTH3_INPUT_FILEPATH)
train_6fold_ground_truth4_input_paths = data_utils.read_paths(TRAIN_6FOLD_GROUND_TRUTH4_INPUT_FILEPATH)
train_6fold_ground_truth5_input_paths = data_utils.read_paths(TRAIN_6FOLD_GROUND_TRUTH5_INPUT_FILEPATH)
train_6fold_ground_truth6_input_paths = data_utils.read_paths(TRAIN_6FOLD_GROUND_TRUTH6_INPUT_FILEPATH)
val_6fold_scan1_input_paths = data_utils.read_paths(VAL_6FOLD_SCAN1_INPUT_FILEPATH)
val_6fold_scan2_input_paths = data_utils.read_paths(VAL_6FOLD_SCAN2_INPUT_FILEPATH)
val_6fold_scan3_input_paths = data_utils.read_paths(VAL_6FOLD_SCAN3_INPUT_FILEPATH)
val_6fold_scan4_input_paths = data_utils.read_paths(VAL_6FOLD_SCAN4_INPUT_FILEPATH)
val_6fold_scan5_input_paths = data_utils.read_paths(VAL_6FOLD_SCAN5_INPUT_FILEPATH)
val_6fold_scan6_input_paths = data_utils.read_paths(VAL_6FOLD_SCAN6_INPUT_FILEPATH)
val_6fold_ground_truth1_input_paths = data_utils.read_paths(VAL_6FOLD_GROUND_TRUTH1_INPUT_FILEPATH)
val_6fold_ground_truth2_input_paths = data_utils.read_paths(VAL_6FOLD_GROUND_TRUTH2_INPUT_FILEPATH)
val_6fold_ground_truth3_input_paths = data_utils.read_paths(VAL_6FOLD_GROUND_TRUTH3_INPUT_FILEPATH)
val_6fold_ground_truth4_input_paths = data_utils.read_paths(VAL_6FOLD_GROUND_TRUTH4_INPUT_FILEPATH)
val_6fold_ground_truth5_input_paths = data_utils.read_paths(VAL_6FOLD_GROUND_TRUTH5_INPUT_FILEPATH)
val_6fold_ground_truth6_input_paths = data_utils.read_paths(VAL_6FOLD_GROUND_TRUTH6_INPUT_FILEPATH)

train_6fold_scan1_output_paths = []
train_6fold_scan2_output_paths = []
train_6fold_scan3_output_paths = []
train_6fold_scan4_output_paths = []
train_6fold_scan5_output_paths = []
train_6fold_scan6_output_paths = []
train_6fold_ground_truth1_output_paths = []
train_6fold_ground_truth2_output_paths = []
train_6fold_ground_truth3_output_paths = []
train_6fold_ground_truth4_output_paths = []
train_6fold_ground_truth5_output_paths = []
train_6fold_ground_truth6_output_paths = []
val_6fold_scan1_output_paths = []
val_6fold_scan2_output_paths = []
val_6fold_scan3_output_paths = []
val_6fold_scan4_output_paths = []
val_6fold_scan5_output_paths = []
val_6fold_scan6_output_paths = []
val_6fold_ground_truth1_output_paths = []
val_6fold_ground_truth2_output_paths = []
val_6fold_ground_truth3_output_paths = []
val_6fold_ground_truth4_output_paths = []
val_6fold_ground_truth5_output_paths = []
val_6fold_ground_truth6_output_paths = []

# Data split input paths
data_split_scan_input_paths = [
    train_trainval_scan_input_paths,
    val_trainval_scan_input_paths,
    train_6fold_scan1_input_paths,
    train_6fold_scan2_input_paths,
    train_6fold_scan3_input_paths,
    train_6fold_scan4_input_paths,
    train_6fold_scan5_input_paths,
    train_6fold_scan6_input_paths,
    val_6fold_scan1_input_paths,
    val_6fold_scan2_input_paths,
    val_6fold_scan3_input_paths,
    val_6fold_scan4_input_paths,
    val_6fold_scan5_input_paths,
    val_6fold_scan6_input_paths
]

data_split_ground_truth_input_paths = [
    train_trainval_ground_truth_input_paths,
    val_trainval_ground_truth_input_paths,
    train_6fold_ground_truth1_input_paths,
    train_6fold_ground_truth2_input_paths,
    train_6fold_ground_truth3_input_paths,
    train_6fold_ground_truth4_input_paths,
    train_6fold_ground_truth5_input_paths,
    train_6fold_ground_truth6_input_paths,
    val_6fold_ground_truth1_input_paths,
    val_6fold_ground_truth2_input_paths,
    val_6fold_ground_truth3_input_paths,
    val_6fold_ground_truth4_input_paths,
    val_6fold_ground_truth5_input_paths,
    val_6fold_ground_truth6_input_paths
]

# Data split output paths
data_split_scan_output_paths = [
    train_trainval_scan_output_paths,
    val_trainval_scan_output_paths,
    train_6fold_scan1_output_paths,
    train_6fold_scan2_output_paths,
    train_6fold_scan3_output_paths,
    train_6fold_scan4_output_paths,
    train_6fold_scan5_output_paths,
    train_6fold_scan6_output_paths,
    val_6fold_scan1_output_paths,
    val_6fold_scan2_output_paths,
    val_6fold_scan3_output_paths,
    val_6fold_scan4_output_paths,
    val_6fold_scan5_output_paths,
    val_6fold_scan6_output_paths
]

data_split_ground_truth_output_paths = [
    train_trainval_ground_truth_output_paths,
    val_trainval_ground_truth_output_paths,
    train_6fold_ground_truth1_output_paths,
    train_6fold_ground_truth2_output_paths,
    train_6fold_ground_truth3_output_paths,
    train_6fold_ground_truth4_output_paths,
    train_6fold_ground_truth5_output_paths,
    train_6fold_ground_truth6_output_paths,
    val_6fold_ground_truth1_output_paths,
    val_6fold_ground_truth2_output_paths,
    val_6fold_ground_truth3_output_paths,
    val_6fold_ground_truth4_output_paths,
    val_6fold_ground_truth5_output_paths,
    val_6fold_ground_truth6_output_paths
]

OUTPUT_DIRECTORIES = [
    TRAIN_TRAINTEST_REF_DIRPATH,
    TEST_TRAINTEST_REF_DIRPATH,
    TRAIN_6FOLD_SCAN_REF_DIRPATH,
    TRAIN_6FOLD_GROUND_TRUTH_REF_DIRPATH,
    VAL_6FOLD_SCAN_REF_DIRPATH,
    VAL_6FOLD_GROUND_TRUTH_REF_DIRPATH
]
def create_output_directories():
    '''
    Set up a sub-directory in data/atlas called atlas_lesion_segmentation that
    has the same directory architecture as atlas_standard but contains data as numpy arrays instead of
    .nii.gz files
    '''
    # Create directory to store numpy data -- als is short for atlas lesion standard
    if not os.path.exists(ATLAS_NUMPY_DATA_DIRPATH):
        os.makedirs(ATLAS_NUMPY_DATA_DIRPATH)

    for dirpath in OUTPUT_DIRECTORIES:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

def scan_to_numpy(path_to_scan):
    '''
    Convert the .nii.gz file given to a numpy array

    Arg(s):
        path_to_scan : str
            path from the current working directory to the .nii.gz file
    Returns:
        numpy : scan
    '''
    scan = nib.load(path_to_scan)
    data = scan.get_fdata()
    return data.astype(np.float32)

def combine_ground_truths(ground_truths):
    '''
    Given list of numpy arrays of ground truths of a scan, combine them into 1 numpy array
    using bitwise OR

    Arg(s):
        ground_truths : list
            list of numpy arrays
    Returns:
        numpy : combines annotated scans from the list ground truths
        if something is invalid (ie: empty list, dimensions of numpy arrays don't match),
            return None
    '''

    if len(ground_truths) == 0:
        # No elements
        return None

    result = ground_truths[0].astype(int)

    for i in range(1, len(ground_truths)):
        if (ground_truths[i].shape != result.shape):
            # If scan shapes don't match
            print('Annotations shape do not match!')
            return None

        result = np.bitwise_or(result, ground_truths[i].astype(int))

    return result

def save_path_end(path):
    '''
    Helper function to make life easier
    Because we are saving numpy arrays in the same directory architecture as atlas_standard,
    we want to save the last 2 directories on the path

    Arg(s):
        path : str
            path to the lowest folder containing the .nii.gz scans
    Returns:
        str : path containing the c00xx/c00xxs00xxtxx of the directory architecture
    '''

    resulting_path, end = os.path.split(path)
    resulting_path, beg = os.path.split(resulting_path)
    return os.path.join(beg, end)

def convert_scans(path_to_dir):
    '''
    This function is for converting the .nii.gz files in the lowest level folder
    It iterates through each scan, converts them to numpy arrays, and combines ground truth scans
    Lastly, it saves the numpy arrays and adds the path for each scan and ground truth to the
        respective global lists

    Arg(s):
        path_to_dir : str
            path to the current directory (data/atlas_standard/c00xx/c00xxs00xxtxx)
    '''

    ground_truths = []
    scan = None
    scan_file_name = ""
    combined_ground_truth_file_name = ""
    dir_contents = os.listdir(path_to_dir)
    # Assumes directory contents are valid
    for item in dir_contents:
        # convert to numpy
        filepath = os.path.join(path_to_dir, item)
        item_scan = scan_to_numpy(filepath)

        if ("LesionSmooth" in item):
            # scan is an ground_truth
            ground_truths.append(item_scan)
        else:
            # Already detected a scan
            if scan is not None:
                print('Detected multiple scans in directory: {}'.format(filepath))
                return None

            scan = item_scan
            # Save output file name
            scan_file_name = item.replace(".nii.gz", ".npy")

    if DEBUG:
        scan_visualization(scan, ground_truths, 100)

    combined_ground_truth = combine_ground_truths(ground_truths)

    # Create file name for combined_ground_truth
    combined_ground_truth_file_name = scan_file_name.replace("t1w", "LesionSmooth")

    # save scans
    save_path = os.path.join(ATLAS_NUMPY_DATA_DIRPATH, save_path_end(path_to_dir))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # define paths to save to
    scan_npy_path = os.path.join(save_path, scan_file_name)
    annotated_npy_path = os.path.join(save_path, combined_ground_truth_file_name)

    # save .npy files
    np.save(scan_npy_path, scan)
    np.save(annotated_npy_path, combined_ground_truth)

    # append paths to the appropriate list
    scan_output_paths.append(scan_npy_path)
    annotated_output_paths.append(annotated_npy_path)

    for idx in range(len(data_split_scan_input_paths)):
        scan_input_split_paths = data_split_scan_input_paths[idx]
        ground_truth_input_split_paths = data_split_ground_truth_input_paths[idx]
        scan_output_split_paths = data_split_scan_output_paths[idx]
        ground_truth_output_split_paths = data_split_ground_truth_output_paths[idx]

        for scan_path, ground_truth_path in zip(scan_input_split_paths, ground_truth_input_split_paths):
            # If both scan and ground truth paths are found in split
            if scan_path in scan_npy_path and ground_truth_path in annotated_npy_path:
                scan_output_split_paths.append(scan_npy_path)
                ground_truth_output_split_paths.append(annotated_npy_path)
                break

def process_partition(partition_path):
    '''
    A partition here is defined to be a folder in atlas_standard with the name of c00XX.
    This function takes in a path to a partition and processes all the scans in the subfolders

    Arg(s):
        partition_path : str
            path to current partition from where function is invoked
    '''
    directories = sorted(os.listdir(partition_path))

    for directory in directories:
        dir_path = os.path.join(partition_path, directory)
        # Avoid trying to access .DS_STORE files like directories
        if os.path.isdir(dir_path):
            convert_scans(dir_path)

def process_data():
    '''
    Iterate through data/atlas/standard and process all .nii.gz files into .npy in every partition
        and directory. Save the lists containing the paths in a .txt file
    '''
    partitions = sorted(os.listdir(ATLAS_RAW_DATA_DIRPATH))

    # process each partition
    for partition in partitions:
        process_partition(os.path.join(ATLAS_RAW_DATA_DIRPATH, partition))
        print("Processed partition " + partition)

    # Save all the paths as text files
    data_utils.write_paths(SCAN_OUTPUT_FILEPATH, scan_output_paths)
    data_utils.write_paths(GROUND_TRUTH_OUTPUT_FILEPATH, annotated_output_paths)

    # Save train val split paths
    data_utils.write_paths(TRAIN_TRAINTEST_SCAN_OUTPUT_FILEPATH, train_trainval_scan_output_paths)
    data_utils.write_paths(TRAIN_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH, train_trainval_ground_truth_output_paths)
    data_utils.write_paths(TEST_TRAINTEST_SCAN_OUTPUT_FILEPATH, val_trainval_scan_output_paths)
    data_utils.write_paths(TEST_TRAINTEST_GROUND_TRUTH_OUTPUT_FILEPATH, val_trainval_ground_truth_output_paths)

    # Save 6-fold split paths
    data_utils.write_paths(TRAIN_6FOLD_SCAN1_OUTPUT_FILEPATH, train_6fold_scan1_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_SCAN2_OUTPUT_FILEPATH, train_6fold_scan2_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_SCAN3_OUTPUT_FILEPATH, train_6fold_scan3_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_SCAN4_OUTPUT_FILEPATH, train_6fold_scan4_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_SCAN5_OUTPUT_FILEPATH, train_6fold_scan5_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_SCAN6_OUTPUT_FILEPATH, train_6fold_scan6_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_GROUND_TRUTH1_OUTPUT_FILEPATH, train_6fold_ground_truth1_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_GROUND_TRUTH2_OUTPUT_FILEPATH, train_6fold_ground_truth2_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_GROUND_TRUTH3_OUTPUT_FILEPATH, train_6fold_ground_truth3_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_GROUND_TRUTH4_OUTPUT_FILEPATH, train_6fold_ground_truth4_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_GROUND_TRUTH5_OUTPUT_FILEPATH, train_6fold_ground_truth5_output_paths)
    data_utils.write_paths(TRAIN_6FOLD_GROUND_TRUTH6_OUTPUT_FILEPATH, train_6fold_ground_truth6_output_paths)
    data_utils.write_paths(VAL_6FOLD_SCAN1_OUTPUT_FILEPATH, val_6fold_scan1_output_paths)
    data_utils.write_paths(VAL_6FOLD_SCAN2_OUTPUT_FILEPATH, val_6fold_scan2_output_paths)
    data_utils.write_paths(VAL_6FOLD_SCAN3_OUTPUT_FILEPATH, val_6fold_scan3_output_paths)
    data_utils.write_paths(VAL_6FOLD_SCAN4_OUTPUT_FILEPATH, val_6fold_scan4_output_paths)
    data_utils.write_paths(VAL_6FOLD_SCAN5_OUTPUT_FILEPATH, val_6fold_scan5_output_paths)
    data_utils.write_paths(VAL_6FOLD_SCAN6_OUTPUT_FILEPATH, val_6fold_scan6_output_paths)
    data_utils.write_paths(VAL_6FOLD_GROUND_TRUTH1_OUTPUT_FILEPATH, val_6fold_ground_truth1_output_paths)
    data_utils.write_paths(VAL_6FOLD_GROUND_TRUTH2_OUTPUT_FILEPATH, val_6fold_ground_truth2_output_paths)
    data_utils.write_paths(VAL_6FOLD_GROUND_TRUTH3_OUTPUT_FILEPATH, val_6fold_ground_truth3_output_paths)
    data_utils.write_paths(VAL_6FOLD_GROUND_TRUTH4_OUTPUT_FILEPATH, val_6fold_ground_truth4_output_paths)
    data_utils.write_paths(VAL_6FOLD_GROUND_TRUTH5_OUTPUT_FILEPATH, val_6fold_ground_truth5_output_paths)
    data_utils.write_paths(VAL_6FOLD_GROUND_TRUTH6_OUTPUT_FILEPATH, val_6fold_ground_truth6_output_paths)


'''
Visualization helper functions
'''
def scan_visualization(np_scan, ground_truths, channel):
    '''
    Utilizes matplotlib to visualize the numpy arrays of scans and ground truths at a particular channel

    Arg(s):
        np_scan : numpy
            the MRI scan as a numpy array
        ground_truths : list[numpy]
            list of the ground truths for this scan as a numpy array
        channel : int
            which channel to visualize
    '''
    num_ground_truths = len(ground_truths)
    comb_annot = combine_ground_truths(ground_truths)

    # create figure
    fig, axs = plt.subplots(2, num_ground_truths)

    # fill in first row with all the ground truths fed in
    if (num_ground_truths > 1):
        for i in range(num_ground_truths):
            axs[0, i].imshow(ground_truths[i][:, :, channel])
            axs[0, i].set_title("ground_truth " + str(i+1) + " out of " + str(num_ground_truths))

    # the second row contains the combined ground_truths with the original scan
    axs[1, 0].imshow(comb_annot[:, :, channel])
    axs[1, 0].set_title("combined ground_truths")
    axs[1, 1].imshow(np_scan[:, :, channel])
    axs[1, 1].set_title("original scan")

    # edit spacing between plots
    fig.tight_layout()

    # show plot
    plt.show()

def visualize_npy(path, channel):
    '''
    This function is to test the output of processing and make sure the saved .npy files are useable
    It reads in folder to .npy files (1 scan & 1 ground truth), loads the scans,
         and displays the scan in channel to matplotlib
    The directory must only contain 2 scans: 1 annotated and 1 unannotated

    Arg(s):
        path : str
            path to the folder to read
        channel : int
            which channel to display in matplotlib
    '''

    # Read in .npy files in the directory
    npy_names = os.listdir(path)
    print(npy_names)
    for name in npy_names:
        if ("LesionSmooth" in name):
            annotated = (np.load(os.path.join(path, name)))
        elif ("t1w" in name):
            # Regular scan
            scan = np.load(os.path.join(path, name))

    # Plot the annotated and normal scan side by side
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(annotated[:, :, channel])
    axs[0].set_title("Annotated Lesions")
    axs[1].imshow(scan[:, :, channel])
    axs[1].set_title("Original Scan")
    fig.tight_layout()
    plt.show()

def visualize_raw(path, channel):
    '''
    This function allows you to visualize the raw scans without storing conversions. Testing purposes only
    Given a path to a directory holding raw .nii.gz scans, convert them to numpy and visualize specific channel
    in matplotlib

    Arg(s):
        path : str
            path to directory with raw scans from PWD
        channel : int
            which channel to display in matplotlib
    '''

    ground_truths = []
    for raw in os.listdir(path):
        npy_arr = scan_to_numpy(os.path.join(path, raw))
        if ("LesionSmooth" in raw):
            ground_truths.append(npy_arr)
        else:
            scan = npy_arr
    scan_visualization(scan, ground_truths, channel)


if __name__ == '__main__':
    create_output_directories()

    # full send :)
    process_data()
