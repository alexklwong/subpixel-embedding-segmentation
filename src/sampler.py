import numpy as np

class PositiveClassSampler():
    '''
    Arg(s):
        positive_class_sample_rates : list[float32]
            positive class sampling rates
        positive_class_sample_schedule : list[float32]
            which epochs to adjust positive_class_sample_rates at
        positive_class_size_thresholds : list[int]
            size of positive class in pixels
    '''
    def __init__(self,
                 positive_class_sample_rates,
                 positive_class_sample_schedule=[-1],
                 positive_class_size_thresholds=[0]):

        assert len(positive_class_sample_rates) > 0, \
            'Invalid length for positive_class_sample_rates'
        assert len(positive_class_sample_schedule) > 0, \
            'Invalid length for positive_class_sample_schedule'
        assert len(positive_class_size_thresholds) > 0, \
            'Invalid length for positive_class_size_thresholds'

        assert len(positive_class_sample_rates) == len(positive_class_sample_schedule) and \
            len(positive_class_sample_rates) == len(positive_class_size_thresholds), \
            'positive_class_sample_rates, positive_class_sample_schedule and positive_class_size_thresholds must have same length'

        for threshold in positive_class_size_thresholds:
            assert threshold >= 0, 'positive_class_size_thresholds contain elements less than 0'

        self.positive_class_sample_rates = positive_class_sample_rates
        self.positive_class_sample_schedule = positive_class_sample_schedule
        self.positive_class_size_thresholds = positive_class_size_thresholds

        self.epoch = 0
        self.positive_sampling_index = 0

        self.positive_sampling_rate = positive_class_sample_rates[self.positive_sampling_index]
        self.positive_class_size_threshold = positive_class_size_thresholds[self.positive_sampling_index]

        self.use_dynamic_schedule = \
            True if -1 not in positive_class_sample_schedule else False

    def random_chunk(self,
                     T,
                     split=True,
                     out_of_range_buffer=0):
        '''
        Perform sampling of the annotation T based on dynamic positive sampling schedule

        Arg(s):
            T : 3D Numpy tensor
                annotation tensor
            split : bool
                whether or not to utilize positive class sampling rate
            out_of_range_buffer : int
                number indicating how many edge chunks are out of bounds
        '''

        all_idx = np.array(range(T.shape[-1]))
        # If we expect to grab a window, then remove the indices on the ends
        if out_of_range_buffer > 0:
            all_idx = all_idx[out_of_range_buffer:-out_of_range_buffer]

        # Case where we have positive sampling rate
        if split:
            # Separate indices of scans with lesions and not
            flags = np.sum(T, axis=(0, 1))

            if out_of_range_buffer > 0:
                flags = flags[out_of_range_buffer:-out_of_range_buffer]

            nonzero_idx = np.nonzero(flags > self.positive_class_size_threshold)[0]
            zero_idx = np.setdiff1d(all_idx, nonzero_idx)

            # Random roll to decide to select scans with lesions (nonzero) or not (zero)
            select_positive_class = \
                zero_idx.shape[0] == 0 \
                or np.random.uniform(low=0.0, high=1.0) <= self.positive_sampling_rate

            if nonzero_idx.shape[0] > 0 and select_positive_class:
                idx = nonzero_idx[np.random.randint(low=0, high=nonzero_idx.shape[0])]
            else:
                idx = zero_idx[np.random.randint(low=0, high=zero_idx.shape[0])]
        else:
            idx = all_idx[np.random.randint(low=0, high=all_idx.shape[0])]

        return idx

    def update_epoch(self):
        '''
        Increases the epoch member variable for sampling schedule and adjust sampling rate
        '''

        self.epoch += 1

        # Case for constant positive class sampling rate
        if -1 in self.positive_class_sample_schedule:
            return

        do_update = \
            self.use_dynamic_schedule and \
            self.epoch > self.positive_class_sample_schedule[self.positive_sampling_index]

        if do_update:
            self.positive_sampling_index += 1
            self.positive_sampling_rate = self.positive_class_sample_rates[self.positive_sampling_index]
            self.positive_class_size_threshold = self.positive_class_size_thresholds[self.positive_sampling_index]
