import cv2
import numpy as np
import os, sys

sys.path.insert(0, 'src')
import data_utils

def has_only_small_lesions(stats,
                           min_lesion_size=0,
                           max_lesion_size=100,
                           verbose=False):
    '''
    Given statistics from cv2.ConnectedComponentsWithStats, return if the scan only contains small lesions

    Arg(s):
        stats : numpy[int]
            Statistics output from cv2.ConnectedComponentsWithStats() (3rd output)
        min_lesion_size : int
            Minimum size (pixels) to be considered a small lesion
        max_lesion_size : int
            Maximum size (pixels) to be considered a small lesion
        verbose : bool
            Verbosity
    Returns:
        bool : whether or not the annotation only contains small lesions
    '''

    # Sort by decreasing size of connected component
    stats = stats[np.argsort(stats[:, -1])[::-1]]

    # Remove background and clean
    sizes = stats[1:, -1]  # all but first row, last column
    if verbose:
        print('Sizes before filtering for noise: \n{}'.format(sizes))

    sizes = sizes[np.where(sizes > min_lesion_size)[0]]
    if verbose:
        print('Sizes: \n{}'.format(sizes))

    # Count the number of large and small lesions
    n_large_lesions = np.where(sizes > max_lesion_size)[0].shape[0]
    n_small_lesions = sizes.shape[0] - n_large_lesions
    if verbose:
        print('{} small lesions and {} large lesions'.format(n_small_lesions, n_large_lesions))

    only_small_lesions = (not n_large_lesions > 0) and n_small_lesions > 0
    return only_small_lesions

def create_2d_small_lesion_map(labels,
                               stats,
                               min_lesion_size=0,
                               max_lesion_size=100,
                               verbose=False):
    '''
    Given a 2D binary annotation of the connectedComponents, return a map of only SMALL lesions

    Arg(s):
        labels : numpy[int]
            H x W connected components map
        stats : numpy[int]
            H x W output stats from cv2.connectedComponentsWithStats
        min_lesion_size : int
            Minimum size (pixels) to be considered a small lesion
        max_lesion_size : int
            Maximum size (pixesl) to be considered a small lesion
        verbose : bool
            Verbosity
    Returns:
        numpy[int] : H x W connected components map of only small lesions
    '''

    unique_ids = np.unique(labels)

    for id in unique_ids:
        lesion_size = stats[id, -1]

        if lesion_size > max_lesion_size or lesion_size < min_lesion_size:
            labels = np.where(labels == id, 0, labels)

    return labels

def small_lesion_2d_processing(ground_truth,
                               min_lesion_size=0,
                               max_lesion_size=100,
                               verbose=False):
    '''
    Given a binary annotation, return information about small lesions

    Arg(s):
        ground_truth : numpy[int]
            Annotation to find connected components from
        min_lesion_size : int
            Minimum size (pixels) to be considered a small lesion
        max_lesion_size : int
            Maximum size (pixesl) to be considered a small lesion
        verbose : bool
            Verbosity
    Returns:
        (numpy[int], bool) :
            * H x W small lesion annotation map
            * whether or not the annotation only contains small lesions
    '''

    # Convert to uint-8 data type
    ground_truth = ground_truth.astype('uint8')

    # Get connected components
    n_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        ground_truth,
        connectivity=8)

    if verbose:
        print("Connected components statistics:\n{}".format(stats))

    # Create small lesion map
    small_lesion_map = create_2d_small_lesion_map(
        labels=output,
        stats=stats,
        min_lesion_size=min_lesion_size,
        max_lesion_size=max_lesion_size,
        verbose=verbose)

    # Determine if this chunk has only small lesions
    only_small_lesions = has_only_small_lesions(
        stats=stats,
        min_lesion_size=min_lesion_size,
        max_lesion_size=max_lesion_size,
        verbose=verbose)

    return small_lesion_map, only_small_lesions

def small_lesion_3d_processing(ground_truth_path,
                               min_lesion_size=0,
                               max_lesion_size=100,
                               save_path=None,
                               verbose=False):
    '''
    Given a path to a 3D MRI ground truth,

    Arg(s):
        ground_truth_path : str
            Path to the 3D Numpy file
        min_lesion_size : int
            Minimum size (pixels) to be considered a small lesion
        max_lesion_size : int
            Maximum size (pixesl) to be considered a small lesion
        save_dir : str or None
            if not None, save the 3d small lesion map to this path
        verbose : bool
            Verbosity
    Returns:
        list[int] : list of chunk indices of this 3D MRI that only contain small lesions
    '''

    # Load ground truth and perform checks
    ground_truth = np.load(ground_truth_path)
    shape = ground_truth.shape
    assert len(shape) == 3

    n_chunks = shape[-1]

    small_lesion_maps = []
    small_lesion_idxs = []

    # For each chunk, perform 2d small lesion map creation
    for chunk_idx in range(n_chunks):
        small_lesion_map, has_only_small_lesions = \
            small_lesion_2d_processing(
                ground_truth=ground_truth[..., chunk_idx],
                min_lesion_size=min_lesion_size,
                max_lesion_size=max_lesion_size,
                verbose=verbose)

        small_lesion_maps.append(small_lesion_map)
        if has_only_small_lesions:
            small_lesion_idxs.append(chunk_idx)

    # Save the small lesion maps
    if save_path is not None:
        small_lesion_maps = np.stack(small_lesion_maps, axis=-1)
        assert small_lesion_maps.shape == shape

        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        np.save(save_path, small_lesion_maps)

    return small_lesion_idxs


def create_small_lesion_maps(ground_truth_paths,
                             min_lesion_size=0,
                             max_lesion_size=100,
                             dir_path=None,
                             ref_path=None,
                             verbose=False):
    '''
    For all the paths in ground_truth_paths, generate
        * small lesion maps
        * which chunks have ONLY small lesions

    Arg(s):
        ground_truth_paths : str
            path to file holding paths to numpy annotations
        min_lesion_size : int
            Minimum size (pixels) to be considered a small lesion
        max_lesion_size : int
            Maximum size (pixesl) to be considered a small lesion
        dir_path : str or None
            if not None, save maps to this directory, maintaining directory structure
        ref_path : str or None
            path to write filepaths to .npy data. If not provided AND dir_path is not None,
                generate in the same directory as annotation_paths. Otherwise use provided value.
        verbose : bool
            Verbosity
    Returns:
        dict : dictionary of MRI ID and the list of indices with only small lesions
    '''

    print("Creating small lesion maps from '{}'...".format(ground_truth_paths))
    ground_truth_paths_list = data_utils.read_paths(ground_truth_paths)
    delim = 'c00'
    small_lesion_dict = {}  # { path (str) : idxs of small lesions (list[int])}
    small_lesion_map_paths = []
    total_small_lesions = 0

    for ground_truth_idx, ground_truth_path in enumerate(ground_truth_paths_list):
        # Create the 3D small lesion maps
        if dir_path is not None:
            local_save_path = delim + ground_truth_path.split(delim, maxsplit=1)[1]
            local_save_path = local_save_path.replace('LesionSmooth_stx', 'small_lesion_map')
            save_path = os.path.join(dir_path, local_save_path)
            small_lesion_map_paths.append(save_path)

            small_lesion_idxs = small_lesion_3d_processing(
                ground_truth_path,
                min_lesion_size=min_lesion_size,
                max_lesion_size=max_lesion_size,
                save_path=save_path,
                verbose=False)
        else:
            small_lesion_idxs = small_lesion_3d_processing(
                ground_truth_path,
                min_lesion_size=min_lesion_size,
                max_lesion_size=max_lesion_size,
                save_path=None,
                verbose=False)

        # Save indices with small lesions
        mri_id = '{}\t{}'.format(
            ground_truth_idx, os.path.basename(os.path.dirname(ground_truth_path)))
        small_lesion_dict[mri_id] = small_lesion_idxs
        total_small_lesions += len(small_lesion_idxs)

    # If save directory is specified, also write the small lesion indices to file
    if dir_path is not None:
        output_dirpath = os.path.dirname(ground_truth_paths)
        output_filepath = os.path.join(
            output_dirpath,
            os.path.basename(ground_truth_paths).replace('ground_truths', 'small_lesion_map_indices'))

        # Save summary of which chunks of which scans have only small lesions
        with open(output_filepath, 'w') as f:
            for ground_truth_path, idxs in small_lesion_dict.items():
                f.write('{}\t{}\t'.format(ground_truth_path, len(idxs)))
                for idx in idxs:
                    f.write('{}\t'.format(idx))
                f.write('\n')
            f.write("--- Summary ---\n")
            f.write("# chunks with only small lesions:\t\t{}".format(total_small_lesions))

        # Save paths to small lesion maps
        if ref_path is None:
            filename = os.path.basename(ground_truth_paths).replace('ground_truths', 'small_lesion_maps')
            ref_path = os.path.join(os.path.dirname(ground_truth_paths), filename)

        print("Saving paths to .npy files at '{}'...".format(ref_path))
        data_utils.write_paths(
            ref_path,
            small_lesion_map_paths)

    print("Success!")
    return small_lesion_dict


if __name__ == "__main__":
    test_ground_truth_paths = \
        os.path.join('testing', 'atlas', 'traintest', 'atlas_test_ground_truths.txt')
    train_ground_truth_paths = \
        os.path.join('training', 'atlas', 'traintest', 'atlas_train_ground_truths.txt')
    save_dir_path = os.path.join('data', 'atlas_spin_small_lesion')

    # Create small lesion maps for ATLAS validation set
    create_small_lesion_maps(
        test_ground_truth_paths,
        dir_path=save_dir_path)
