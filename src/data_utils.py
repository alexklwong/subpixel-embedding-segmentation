import os
import numpy as np
import nibabel as nib
import torch


def read_paths(filepath):
    '''
    Stores a depth map into an image (16 bit PNG)

    Arg(s):
        path : str
            path to file where data will be stored
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def save_numpy_to_nii(np_arr, path):
    '''
    Convert a numpy array into an .nii file and save to path

    Arg(s):
        np_arr : numpy
            numpy data to convert to .nii
        path : str
            Path to save .nii file to (including filename)
    '''

    # Check for valid filename
    assert path[-4:] == '.nii'

    # Convert to nii and save
    nii_img = nib.nifti1.Nifti1Image(np_arr, affine=np.eye(4))
    nib.save(nii_img, path)

def save_annotation(annotation, save_dir, smir_id, description='annotation'):
    '''
    Save an annotation in the format specified by SMIR

    Arg(s):
        annotation : numpy[float32]
            Annotation to save
        save_dir : str
            Directory to save file in
        smir_id : str
            ID number that this annotation corresponds to
        description : str
            description for the file
    '''

    # Create filename in specified format
    filename = 'SMIR.{}.{}.nii'.format(description, smir_id)
    # If necessary, create output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)
    save_numpy_to_nii(annotation, save_path)

def get_dataset_stats(path):
    '''
    Find the minimum, maximum, mean, and standard deviation of the entire dataset

    Arg(s):
        path : str
            path the the .txt file containing all scan file paths
    Returns:
        dict[str, float32] : {'min': min, 'max': max, 'mean': mean, 'stddev': stddev}
    '''

    scan_paths = read_paths(path)
    scans_array = []

    # Load each MRI scan into the array scans_array
    for scan_path in scan_paths:
        scans_array.append(np.load(scan_path).flatten())

    # Concatenate all scans in one numpy array
    all_scans = np.concatenate(scans_array, axis=-1)

    # Find min, max, mean and standard deviation
    dataset_min = np.amin(all_scans)
    dataset_max = np.amax(all_scans)
    dataset_mean = np.mean(all_scans)
    dataset_stddev = np.std(all_scans)

    return {
        'min': dataset_min,
        'max': dataset_max,
        'mean': dataset_mean,
        'stddev': dataset_stddev
    }

def min_max_normalization(T, dataset_min, dataset_max):
    '''
    Normalize the input by its minimum and maximum values

    Arg(s):
        T : numpy[float32]
            array to normalize
        dataset_min : float32
            minimum value of dataset
        dataset_max : float32
            maximum value of dataset
    Returns:
        numpy[float32] : normalized array
    '''

    return ((T - dataset_min) / (dataset_max - dataset_min))

def parse_small_lesion_txt(path, n_samples=None):
    '''
    Given a path to a .txt file saved in create_small_lesion_maps(),
        return list of list of idxs of small indices

    Arg(s):
        path: str
            List to .txt file saved in the format above
        n_samples : int or None
            Optional argument used as a sanity check to make sure we have the correct number
            of idx lists
    Returns:
        list[list[int]] : each outer list corresponds with a patient scan.
        The order corresponds with the order in the .txt file
            Elements of inner list correspond with chunk idx of ONLY small lesions
    '''

    lines = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            idx, mri_id, n_small_lesions = line[0:3]

            small_lesion_idxs = line[3:]
            if len(small_lesion_idxs) == 0:
                continue
            try:
                small_lesion_idxs = [int(idx) for idx in small_lesion_idxs]
            except ValueError:
                continue
            set_small_lesion_idxs = set(small_lesion_idxs)
            assert len(set_small_lesion_idxs) == len(small_lesion_idxs)

            lines.append(set_small_lesion_idxs)

    if n_samples is not None:
        assert n_samples == len(lines)
    return lines

def get_n_chunk(scan,
                gt_scan_id,
                n_chunk,
                is_torch=False,
                input_type='BDHWC',
                constants=[0, 0, 0, 0, 0]):
    '''
    Arg(s):
        scan : torch.tensor[float32] or numpy[float32]
            patient scan
        gt_scan_id : int
            scan id within the patient scan
        n_chunk : int
            number of scans in the 3D slice
        is_torch : bool
            if True, scan should be torch tensor
        input_type : str
            BDHWC, DHWC, or BCDHW
        constants : list[int]
            values to pad each modality by, len(constants) = D
    Returns:
        numpy[float32] : n_chunk 2D slices from scan
    '''

    scan_shape = scan.shape
    chunk_range = n_chunk // 2

    if input_type == 'BCDHW' and is_torch:
        # B x c x D x H x W
        temp_shape = (scan_shape[0], n_chunk, *scan_shape[2:])
    elif input_type == 'BDHWC' or input_type == 'DHWC':
        # B x D x H x W x c or D x H x W x c
        temp_shape = (*scan_shape[:-1], n_chunk)
    else:
        raise NotImplementedError

    scan_temp = constant_padding_template(
        shape=temp_shape,
        constants=constants,
        is_torch=is_torch,
        input_type=input_type)

    if is_torch and input_type in ['BCDHW']:  # used in validate
        num_scans = scan.shape[1]
    elif not is_torch and input_type in ['BDHWC', 'DHWC']:  # used in datasets
        num_scans = scan.shape[-1]
    else:
        raise NotImplementedError

    # scan_temp = zeros with padding
    # scan = actual scan
    # Index into scan
    index_mid = gt_scan_id
    index_left = max(0, index_mid - chunk_range)
    # Compare to num_scans - 1 bc we add 1 later when indexing
    index_right = min(index_mid + chunk_range, num_scans-1)

    index_mid_temp = n_chunk // 2
    range_left_temp = index_mid - index_left
    range_right_temp = index_right - index_mid

    if input_type in ['BCDHW']:
        scan_temp[:, index_mid_temp-range_left_temp:index_mid_temp+range_right_temp+1, ...] = \
            scan[:, index_left:index_right+1, ...]

    elif input_type in ['BDHWC', 'DHWC']:
        scan_temp[..., index_mid_temp-range_left_temp:index_mid_temp+range_right_temp+1] = \
            scan[..., index_left:index_right+1]

    return scan_temp

def constant_padding_template(shape,
                              constants=[0, 0, 0, 0, 0],
                              is_torch=False,
                              input_type='BDHWC'):
    '''
    Returns a temporary tensor/array filled with constants

    Arg(s):
        scan : torch.tensor[float32] or numpy[float32]
            patient scan
        constants : list[float32]
            D-dimension array of constants by which to pad each modality by
        is_torch : bool
            if True, scan should be torch tensor
        input_type : str
            'BDHWC', 'BCDHW', 'CDHW', 'DHWC', 'CHW', or 'HWC'
    Returns:
        torch.tensor[float32] or numpy[float32] : array filled array with constant values
    '''

    if is_torch:
        ones = torch.ones
        constants = torch.tensor(constants, dtype=torch.float)
    else:
        ones = np.ones
        constants = np.array(constants, dtype=np.float32)

    if input_type == 'BCDHW' and is_torch:
        assert len(constants) == shape[2]  # Assert a constant for each modality
        constants = constants.view((1, 1, -1, 1, 1))  # Resahpe constants to be 1 x 1 x D x 1 x 1

    elif input_type == 'BDHWC':
        assert len(constants) == shape[1]  # Assert a constant for each modality
        constants = constants.reshape((1, -1, 1, 1, 1))

    elif input_type == 'DHWC':
        assert len(constants) == shape[0]  # Assert a constant for each modality
        constants = constants.reshape((-1, 1, 1, 1))

    elif input_type == 'CDHW':
        assert len(constants) == shape[1]  # Assert a constant for each modality
        constants = constants.reshape((1, -1, 1, 1))

    elif input_type == 'CHW' or input_type == 'HWC':
        assert len(constants) == 1

    else:
        raise NotImplementedError

    scan_temp = ones(shape) * constants

    return scan_temp
