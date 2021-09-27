import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def create_output_directories(directories=[]):
    '''
    Create all directories provided in directories

    Arg(s):
        directories: list of str
    '''

    # If directories is list is empty, return
    if not isinstance(directories, list) or not len(directories):
        return

    for directory in directories:

        if not isinstance(directory, str):
            raise ValueError('Invalid path to directory')

        if not os.path.exists(directory):
            os.makedirs(directory)


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

def atlas_visualize_npy(path, channel):
    '''
    This function is to test the output of processing and make sure the saved .npy files are useable

    It reads in folder to .npy files (1 scan & 1 annotation), loads the scans,
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

def append_path_prefix(paths, prefix):
    '''
    For each path in paths, append the prefix to the path to create local paths to data

    Arg(s):
        paths : list[str]
            paths to where numpy data is located
        prefix : str
            prefix to data directory
    Returns:
        list[str] : paths with prefix
    '''

    paths = [os.path.join(prefix, path) for path in paths]
    return paths
