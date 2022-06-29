import numpy as np
import torch.utils.data
import data_utils


class SPiNMRITrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for Subpixel Network (SPiN) to fetch
    (1) input scans (supports multimodal)
    (2) ground truth annotations

    Args:
        multimodal_scan_paths : list[list[str]]
            list of list of paths to multiple imaging modalities
        ground_truth_paths : list[str]
            list of paths to ground truth annotations
        shape : tuple[int]
            (n_chunk, n_height, n_width) tuple
        padding_constants : list[int]
            Value to pad with for selecting scans
        positive_class_sampler : Sampler
            sampler class instance to pick data with scheduled positive class rate
    '''
    def __init__(self,
                 multimodal_scan_paths,
                 ground_truth_paths,
                 shape,
                 padding_constants=[0, 0, 0, 0, 0],
                 # Data sampling bias
                 positive_class_sampler=None):

        # Make sure that input is list of list of strings
        n_sample = -1

        assert isinstance(multimodal_scan_paths, list)

        for scan_paths in multimodal_scan_paths:
            assert isinstance(scan_paths, list)
            assert len(scan_paths) > 0
            assert isinstance(scan_paths[0], str)

            if n_sample == -1:
                n_sample = len(scan_paths)
            else:
                assert len(scan_paths) == n_sample

        # Dataset paths and shape
        self.multimodal_scan_paths = multimodal_scan_paths
        self.n_sample = len(multimodal_scan_paths[0])

        self.n_chunk = shape[0]
        self.n_height = shape[1]
        self.n_width = shape[2]

        self.ground_truth_paths = ground_truth_paths

        # Augmentation settings
        self.positive_class_sampler = positive_class_sampler

        self.padding_constants = padding_constants

    def __getitem__(self, index):
        '''
        Fetches scan and ground truth annotation

        Arg(s):
            index : int
                index of sample in dataset
        Returns
            numpy[float32] : C x D x H x W input scan
            numpy[uint64] : 1 x H x W ground truth annotations
        '''

        # List of input scans
        scans = []

        for idx, scan_paths in enumerate(self.multimodal_scan_paths):

            # Each scan is H x W x C
            scan_path = scan_paths[index]
            scan = np.load(scan_path).astype(np.float32)

            # Append scan to list of scans
            scans.append(scan)

        # Concatenate scans together into D x H x W x C
        scan = np.stack(scans, axis=0)

        # Each scan is H x W x C
        ground_truth = np.load(self.ground_truth_paths[index]).astype(np.float32)

        # Account for noise and normalize
        ground_truth = np.where(ground_truth > 0, 1, 0)

        if self.n_chunk is not None:
            # Randomly select a chunk
            selected_idx = self.positive_class_sampler.random_chunk(
                T=ground_truth,
                split=True)

            if self.n_chunk > 1:
                # Select the chunk from the scan volume and if out of bounds, then pad
                scan = data_utils.get_n_chunk(
                    scan,
                    selected_idx,
                    self.n_chunk,
                    constants=self.padding_constants,
                    input_type='DHWC')

                # Scan shape: D x H x W x c -> c x D x H x W
                scan = np.transpose(scan, (3, 0, 1, 2))
            else:
                # Scan shape: D x H x W -> 1 x D x H x W
                scan = np.expand_dims(scan[..., selected_idx], 0)

            # Ground truth shape: H x W -> 1 x H x W
            ground_truth = np.expand_dims(ground_truth[..., selected_idx], 0)
        else:
            # Scan shape: D x H x W x C -> C x D x H x W
            scan = np.transpose(scan, (3, 0, 1, 2))

            # Ground truth shape: H x W x C -> C x H x W
            ground_truth = np.transpose(ground_truth, (2, 0, 1))

        scan_data_format = 'CHW' if len(scan.shape) == 3 else 'CDHW'
        ground_truth_data_format = 'CHW'

        # Resize from C x D x H x W to C x D x h x w
        do_resize = \
            self.n_height is not None and \
            self.n_width is not None and \
            scan.shape[2] != self.n_height and \
            scan.shape[3] != self.n_width

        if do_resize:
            scan = data_utils.resize(
                scan,
                shape=(self.n_height, self.n_width),
                interp_type='nearest',
                data_format=scan_data_format)

            ground_truth = data_utils.resize(
                ground_truth,
                shape=(self.n_height, self.n_width),
                interp_type='nearest',
                data_format=ground_truth_data_format)

        return scan.astype(np.float32), ground_truth.astype(np.int64)

    def __len__(self):
        return self.n_sample


class SPiNMRIInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for Subpixel Network (SPiN) to fetch entire scan volumes (supports multimodal)

    Args:
        multimodal_scan_paths : list[list[str]]
            list of list of paths to multiple imaging modalities
        shape : tuple[int]
            (n_height, n_width) tuple
    '''
    def __init__(self, multimodal_scan_paths, shape):

        # Make sure that input is list of list of strings
        n_sample = -1

        assert isinstance(multimodal_scan_paths, list)

        for scan_paths in multimodal_scan_paths:
            assert isinstance(scan_paths, list)
            assert len(scan_paths) > 0
            assert isinstance(scan_paths[0], str)

            if n_sample == -1:
                n_sample = len(scan_paths)
            else:
                assert len(scan_paths) == n_sample

        # Dataset paths and shape
        self.multimodal_scan_paths = multimodal_scan_paths
        self.n_sample = len(multimodal_scan_paths[0])

        self.n_height = shape[0]
        self.n_width = shape[1]

    def __getitem__(self, index):
        '''
        Fetches entire scan volume

        Arg(s):
            index : int
                index of sample in dataset
        Returns
            numpy[float32] : C x D x H x W input scan
        '''

        # List of input scans
        scans = []

        for idx, scan_paths in enumerate(self.multimodal_scan_paths):

            # Each scan is H x W x C
            scan_path = scan_paths[index]
            scan = np.load(scan_path).astype(np.float32)

            # Append scan to list of scans
            scans.append(scan)

        # Concatenate scans together into D x H x W x C
        scan = np.stack(scans, axis=0)

        # Scan shape: D x H x W x C -> C x D x H x W
        scan = np.transpose(scan, (3, 0, 1, 2))

        scan_data_format = 'CHW' if len(scan.shape) == 3 else 'CDHW'

        # Resize from C x D x H x W to C x D x h x w
        do_resize = \
            self.n_height is not None and \
            self.n_width is not None and \
            scan.shape[2] != self.n_height and \
            scan.shape[3] != self.n_width

        if do_resize:
            scan = data_utils.resize(
                scan,
                shape=(self.n_height, self.n_width),
                interp_type='nearest',
                data_format=scan_data_format)

        return scan.astype(np.float32)

    def __len__(self):
        return self.n_sample

class SPiNRGBTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for Subpixel Network (SPiN) to fetch
    (1) input RGB scans
    (2) ground truth annotations

    Args:
        scan_paths : list[str]
            list of paths to RGB images
        ground_truth_paths : list[str]
            list of paths to annotations
        shape : tuple[int]
            (n_height, n_width) tuple
        padding_constant : int
            value to pad by
    '''
    def __init__(self,
                 scan_paths,
                 ground_truth_paths,
                 shape,
                 padding_constant=0):

        # Make sure that input is list of list of strings

        assert isinstance(scan_paths, list)
        self.n_sample = len(scan_paths)

        # Dataset paths and shape
        self.scan_paths = scan_paths

        # Shape check
        assert len(shape) == 2
        self.n_height = shape[0]
        self.n_width = shape[1]

        if ground_truth_paths is not None:
            self.ground_truth_paths = ground_truth_paths
        else:
            self.ground_truth_paths = [None] * self.n_sample

        # Augmentation settings
        self.padding_constant = padding_constant

    def __getitem__(self, index):
        '''
        Fetches scan(s) (and annotation if available)
        Args:
            index : int
                index of sample in dataset
        Returns
            (scan, annotation) : tuple[numpy]
                tuple of numpy arrays of scan and annotations
        '''

        # Load in scan
        scan = np.load(self.scan_paths[index]).astype(np.float32)  # Loads as H x W x 3

        # Each scan is H x W x C
        annotation = np.load(self.ground_truth_paths[index]).astype(np.float32)

        # Account for noise and normalize
        annotation = np.where(annotation > 0, 1, 0)
        # Expand dims for annotation
        annotation = np.expand_dims(annotation, axis=0)

        # Scan shape: H x W x C -> C x H x W
        scan = np.transpose(scan, (2, 0, 1))

        scan_data_format = 'CHW' if len(scan.shape) == 3 else 'CDHW'
        annotation_data_format = 'CHW'

        # Resize from C x H x W to C x h x w
        do_crop = \
            self.n_height is not None and \
            self.n_width is not None and \
            scan.shape[1] > self.n_height and \
            scan.shape[2] > self.n_width

        if do_crop:
            x_top_left = np.random.randint(0, scan.shape[2] - self.n_width)
            y_top_left = np.random.randint(0, scan.shape[1] - self.n_height)

            scan = data_utils.crop(
                T=scan,
                top_left=(y_top_left, x_top_left),
                shape=(self.n_height, self.n_width),
                data_format=scan_data_format)

            annotation = data_utils.crop(
                T=annotation,
                top_left=(y_top_left, x_top_left),
                shape=(self.n_height, self.n_width),
                data_format=scan_data_format)

        return (scan.astype(np.float32), annotation.astype(np.int64)) #, small_lesion_map.astype(np.int64))

    def __len__(self):
        return self.n_sample


class SPiNRGBInferenceDataset(torch.utils.data.Dataset):
    '''
    Args:
        scan_paths : list[str]
            list of paths to RGB images
        shape : tuple[int]
            (n_height, n_width) tuple
    '''
    def __init__(self, scan_paths, shape):

        # Make sure that input is list of list of strings

        assert isinstance(scan_paths, list)
        self.n_sample = len(scan_paths)

        # Dataset paths and shape
        self.scan_paths = scan_paths

        # Shape check
        assert len(shape) == 2
        self.n_height = shape[0]
        self.n_width = shape[1]


    def __getitem__(self, index):
        '''
        Fetches scan(s) (and annotation if available)
        Args:
            index : int
                index of sample in dataset
        Returns
            numpy[float32] : C x H x W input image scan
        '''

        # Load in scan
        scan = np.load(self.scan_paths[index]).astype(np.float32)  # Loads as H x W x 3

        # Scan shape: H x W x C -> C x H x W
        scan = np.transpose(scan, (2, 0, 1))

        scan_data_format = 'CHW' if len(scan.shape) == 3 else 'CDHW'

        # Resize from C x H x W to C x h x w
        do_crop = \
            self.n_height is not None and \
            self.n_width is not None and \
            scan.shape[1] > self.n_height and \
            scan.shape[2] > self.n_width

        if do_crop:
            x_top_left = np.random.randint(0, scan.shape[2] - self.n_width)
            y_top_left = np.random.randint(0, scan.shape[1] - self.n_height)

            scan = data_utils.crop(
                T=scan,
                top_left=(y_top_left, x_top_left),
                shape=(self.n_height, self.n_width),
                data_format=scan_data_format)

        return scan.astype(np.float32)

    def __len__(self):
        return self.n_sample