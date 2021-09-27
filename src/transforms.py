import torch
import torchvision.transforms.functional as functional
from PIL import Image
import numpy as np


class Transforms(object):

    def __init__(self,
                 dataset_means=[-1, -1, -1],
                 dataset_stddevs=[-1, -1, -1],
                 dataset_normalization='standard',
                 pad_to_shape=[-1, -1],
                 random_crop_to_shape=[-1, -1],
                 random_flip_type=['none'],
                 random_rotate=-1,
                 random_intensity=[-1, -1],
                 random_noise_type='none',
                 random_noise_spread=-1,
                 random_crop_and_pad=[-1, -1],
                 random_resize_and_pad=[-1, -1]):
        '''
        Transforms and augmentation class

        Args:
            dataset_means : list[float]
                per channel mean of dataset
            dataset_stddevs : list[float]
                per channel standard deviation of dataset
            dataset_normalization : str
                method to normalize images
            pad_to_shape : list[int]
                output shape after transforms
            random_crop_to_shape : list[int]
                output shape after random crop
            random_flip_type : list[str]
                none, horizontal, vertical
            random_intensity : list[int]
                intensity adjustment [0, B], from 0 (black image) to B factor increase
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
            random_rotate : float
                symmetric min and max amount to rotate
            random_crop_and_pad : list[int]
                min and max percentage to crop for random crop and pad
            random_resize_and_pad : list[int]
                min and max percentage to resize for random resize and pad
        '''

        self.dataset_means = dataset_means
        self.dataset_stddevs = dataset_stddevs

        self.dataset_normalization = dataset_normalization

        self.do_pad_to_shape = True if -1 not in pad_to_shape else False
        self.pad_to_shape_height = pad_to_shape[0]
        self.pad_to_shape_width = pad_to_shape[1]

        # Intensity augmentations
        self.do_random_intensity = True if -1 not in random_intensity else False
        self.random_intensity = random_intensity

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > 0) else False

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

        # Geometric augmentations
        self.do_random_crop_to_shape = True if -1 not in random_crop_to_shape else False
        self.random_crop_to_shape_height = random_crop_to_shape[0]
        self.random_crop_to_shape_width = random_crop_to_shape[1]

        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

        self.do_random_rotate = True if random_rotate > 0 else False
        self.random_rotate = random_rotate

        self.do_random_crop_and_pad = True if -1 not in random_crop_and_pad else False

        self.random_crop_and_pad_min = random_crop_and_pad[0]
        self.random_crop_and_pad_max = random_crop_and_pad[1]

        if self.do_random_crop_and_pad:
            assert self.random_crop_and_pad_min < self.random_crop_and_pad_max
            assert self.random_crop_and_pad_max <= 1

        self.do_random_resize_and_pad = True if -1 not in random_resize_and_pad else False

        self.random_resize_and_pad_min = random_resize_and_pad[0]
        self.random_resize_and_pad_max = random_resize_and_pad[1]

        if self.do_random_resize_and_pad:
            assert self.random_resize_and_pad_min < self.random_resize_and_pad_max
            assert self.random_resize_and_pad_min > 0

    def transform(self, images_arr, labels_arr=[], random_transform_probability=0.50):
        '''
        Applies transform to images and ground truth

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            labels_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
            list[torch.Tensor] : list of transformed N x c x H x W label tensors
        '''

        device = images_arr[0].device
        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        elif n_dim == 5:
            n_batch, _, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            torch.rand(n_batch, device=device) <= random_transform_probability

        # Convert all images to float
        images_arr = [
            images.float() for images in images_arr
        ]

        # Normalize images
        images_arr = self.normalize_images(
            images_arr,
            normalization=self.dataset_normalization,
            means=self.dataset_means,
            stddevs=self.dataset_stddevs)

        # Intensity augmentations are applied to only images
        if self.do_random_intensity:

            do_adjust_intensity = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            intensity_min, intensity_max = self.random_intensity
            factors = (intensity_max - intensity_min) * values + intensity_min

            images_arr = self.adjust_intensity(images_arr, do_adjust_intensity, factors)

        if self.do_random_noise:

            do_add_noise = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.add_noise(
                images_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        # Geometric transformations are applied to both images and ground truths
        if self.do_random_crop_to_shape and torch.rand(1, device=device) <= 0.50:

            # Random crop factors
            start_y = torch.randint(
                low=0,
                high=n_height - self.random_crop_to_shape_height,
                size=(n_batch,),
                device=device)

            start_x = torch.randint(
                low=0,
                high=n_width - self.random_crop_to_shape_width,
                size=(n_batch,),
                device=device)

            end_y = start_y + self.random_crop_to_shape_height
            end_x = start_x + self.random_crop_to_shape_width

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            images_arr = self.crop(
                images_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            labels_arr = self.crop(
                labels_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            # Update shape of tensors after crop
            n_height = self.random_crop_to_shape_height
            n_width = self.random_crop_to_shape_width

        if self.do_random_horizontal_flip:

            do_horizontal_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            labels_arr = self.horizontal_flip(
                labels_arr,
                do_horizontal_flip)

        if self.do_random_vertical_flip:

            do_vertical_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            labels_arr = self.vertical_flip(
                labels_arr,
                do_vertical_flip)

        if self.do_random_rotate:

            do_rotate = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = np.random.rand(n_batch)

            rotate_min = -self.random_rotate
            rotate_max = self.random_rotate

            angles = (rotate_max - rotate_min) * values + rotate_min

            images_arr = self.rotate(images_arr, do_rotate, angles)

            labels_arr = self.rotate(labels_arr, do_rotate, angles)

        if self.do_random_resize_and_pad:

            do_resize_and_pad = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            # Random resize factors
            r_height = torch.randint(
                low=int(self.random_resize_and_pad_min * n_height),
                high=int(self.random_resize_and_pad_max * n_height),
                size=(n_batch,),
                device=device)

            r_width = torch.randint(
                low=int(self.random_resize_and_pad_min * n_width),
                high=int(self.random_resize_and_pad_max * n_width),
                size=(n_batch,),
                device=device)

            shape = [r_height, r_width]

            # Random padding along all sizes
            d_height = (n_height - r_height).int()
            pad_top = (d_height * torch.rand(n_batch, device=device)).int()
            pad_bottom = d_height - pad_top

            d_width = (n_width - r_width).int()
            pad_left = (d_width * torch.rand(n_batch, device=device)).int()
            pad_right = d_width - pad_left

            pad_top = torch.maximum(pad_top, torch.full_like(pad_top, fill_value=0))
            pad_bottom = torch.maximum(pad_bottom, torch.full_like(pad_bottom, fill_value=0))
            pad_left = torch.maximum(pad_left, torch.full_like(pad_left, fill_value=0))
            pad_right = torch.maximum(pad_right, torch.full_like(pad_right, fill_value=0))

            padding = [pad_top, pad_bottom, pad_left, pad_right]

            images_arr = self.resize_and_pad(
                images_arr,
                do_resize_and_pad=do_resize_and_pad,
                resize_shape=shape,
                max_shape=(n_height, n_width),
                padding=padding,
                padding_mode='edge')

            labels_arr = self.resize_and_pad(
                labels_arr,
                do_resize_and_pad=do_resize_and_pad,
                resize_shape=shape,
                max_shape=(n_height, n_width),
                padding=padding,
                padding_mode='edge')

        if self.do_random_crop_and_pad:

            do_crop_and_pad = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            # Random crop factors
            max_height = int(self.random_crop_and_pad_max * n_height)
            min_height = int(self.random_crop_and_pad_min * n_height)
            max_width = int(self.random_crop_and_pad_max * n_width)
            min_width = int(self.random_crop_and_pad_min * n_width)

            start_y = torch.randint(
                low=0,
                high=max_height - min_height,
                size=(n_batch,),
                device=device)

            start_x = torch.randint(
                low=0,
                high=max_width - min_width,
                size=(n_batch,),
                device=device)

            end_y = start_y + torch.randint(
                low=min_height,
                high=max_height,
                size=(n_batch,),
                device=device)

            end_x = start_x + torch.randint(
                low=min_width,
                high=max_width,
                size=(n_batch,),
                device=device)

            end_y = torch.minimum(end_y, torch.full_like(end_y, fill_value=n_height))
            end_x = torch.minimum(end_x, torch.full_like(end_x, fill_value=n_width))

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            # Random padding along all sizes
            d_height = (n_height - (end_y - start_y)).int()
            pad_top = (d_height * torch.rand(n_batch, device=device)).int()
            pad_bottom = d_height - pad_top

            d_width = (n_width - (end_x - start_x)).int()
            pad_left = (d_width * torch.rand(n_batch, device=device)).int()
            pad_right = d_width - pad_left

            padding = [pad_top, pad_bottom, pad_left, pad_right]

            images_arr = self.crop_and_pad(
                images_arr,
                do_crop_and_pad=do_crop_and_pad,
                start_yx=start_yx,
                end_yx=end_yx,
                padding=padding,
                padding_mode='edge')

            labels_arr = self.crop_and_pad(
                labels_arr,
                do_crop_and_pad=do_crop_and_pad,
                start_yx=start_yx,
                end_yx=end_yx,
                padding=padding,
                padding_mode='edge')

        if self.do_pad_to_shape:

            images_arr = self.pad_to_shape(
                images_arr,
                shape=(self.pad_to_shape_height, self.pad_to_shape_width),
                padding_mode='constant',
                padding_value=0)

            labels_arr = self.pad_to_shape(
                labels_arr,
                shape=(self.pad_to_shape_height, self.pad_to_shape_width),
                padding_mode='constant',
                padding_value=0)

        # Return the transformed inputs
        if len(labels_arr) == 0:
            return images_arr
        else:
            return images_arr, labels_arr

    def normalize_images(self,
                         images_arr,
                         normalization='none',
                         means=[-1],
                         stddevs=[-1]):
        '''
        Normalize images

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            normalization : str
                standard, none
            means : list[float]
                per channel mean
            stddevs : list[float]
                per channel standard deviation
        Returns:
            images_arr : list of normalized N x C x H x W tensors
        '''

        if normalization == 'standard':
            n_dim = images_arr[0].ndim
            device = images_arr[0].device

            means = torch.tensor(means, dtype=torch.float32, device=device)
            stddevs = torch.tensor(stddevs, dtype=torch.float32, device=device)
            if n_dim == 4:
                means = means.view(1, -1, 1, 1)
                stddevs = stddevs.view(1, -1, 1, 1)
            elif n_dim == 5:
                means = means.view(1, -1, 1, 1, 1)
                stddevs = stddevs.view(1, -1, 1, 1, 1)

            for i, images in enumerate(images_arr):
                images_arr[i] = (images - means) / stddevs

        elif normalization == 'none':
            pass
        else:
            raise ValueError('Unsupported normalization type: {}'.format(
                normalization))

        return images_arr

    def pad_to_shape(self, images_arr, shape, padding_mode='constant', padding_value=0):
        '''
        Pads images to shape

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            shape : list[int]
                output shape after padding
            padding_mode : str
                mode for padding: constant, edge, reflect
            padding_value : float
                value to pad with
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        p_height, p_width = shape

        for i, images in enumerate(images_arr):

            n_height, n_width = images.shape[-2:]
            d_height = max(p_height - n_height, 0)
            d_width = max(p_width - n_width, 0)

            if d_height > 0 or d_width > 0:

                pad_top = d_height // 2
                pad_bottom = d_height - pad_top
                pad_left = d_width // 2
                pad_right = d_width - pad_left

                images = functional.pad(
                    images,
                    (pad_left, pad_top, pad_right, pad_bottom),
                    padding_mode=padding_mode,
                    fill=padding_value)

            images_arr[i] = images

        return images_arr

    def crop(self, images_arr, start_yx, end_yx):
        '''
        Performs on on images
        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            images_cropped = []

            for b, image in enumerate(images):

                start_y = start_yx[0][b]
                start_x = start_yx[1][b]
                end_y = end_yx[0][b]
                end_x = end_yx[1][b]

                # Crop image
                image = image[..., start_y:end_y, start_x:end_x]

                images_cropped.append(image)

            images_arr[i] = torch.stack(images_cropped, dim=0)

        return images_arr

    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = functional.hflip(image)

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = functional.vflip(image)

            images_arr[i] = images

        return images_arr

    def rotate(self, images_arr, do_rotate, angles):
        '''
        Rotates each sample

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_rotate : bool
                N booleans to determine if rotation is performed on each sample
            angles : float
                N floats to determine how much to rotate each sample
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_rotate[b]:
                    images[b, ...] = functional.rotate(
                        image,
                        angle=angles[b],
                        resample=Image.NEAREST,
                        expand=False)

            images_arr[i] = images

        return images_arr

    def resize_and_pad(self,
                       images_arr,
                       do_resize_and_pad,
                       resize_shape,
                       max_shape,
                       padding,
                       padding_mode='constant',
                       padding_value=0):
        '''
        Resize and pad images

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_resize_and_pad : bool
                N booleans to determine if image will be resized and padded
            resize_shape : list[int, int]
                height and width to resize
            max_shape : tuple[int]
                max height and width, if exceed center crop
            padding : list[int, int, int, int]
                list of padding for top, bottom, left, right sides
            padding_mode : str
                mode for padding: constant, edge, reflect
            padding_value : float
                value to pad with
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_resize_and_pad[b]:
                    image = images[b, ...]

                    r_height = resize_shape[0][b]
                    r_width = resize_shape[1][b]
                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]

                    # Resize image
                    image = functional.resize(
                        image,
                        size=(r_height, r_width),
                        interpolation=Image.NEAREST)

                    # Pad image
                    image = functional.pad(
                        image,
                        (pad_left, pad_top, pad_right, pad_bottom),
                        padding_mode=padding_mode,
                        fill=padding_value)

                    height, width = image.shape[-2:]
                    max_height, max_width = max_shape

                    # If resized image is larger, then do center crop
                    if max_height < height or max_width < width:
                        start_y = height - max_height
                        start_x = width - max_width
                        end_y = start_y + max_height
                        end_x = start_x + max_width
                        image = image[..., start_y:end_y, start_x:end_x]

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def crop_and_pad(self,
                     images_arr,
                     do_crop_and_pad,
                     start_yx,
                     end_yx,
                     padding,
                     padding_mode='constant',
                     padding_value=0):
        '''
        Crop and pad images

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_crop_and_pad : bool
                N booleans to determine if image will be cropped and padded
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
            padding : list[int, int, int, int]
                list of padding for top, bottom, left, right sides
            padding_mode : str
                mode for padding: constant, edge, reflect
            padding_value : float
                value to pad with
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_crop_and_pad[b]:
                    image = images[b, ...]

                    start_y = start_yx[0][b]
                    start_x = start_yx[1][b]
                    end_y = end_yx[0][b]
                    end_x = end_yx[1][b]
                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]

                    # Crop image
                    image = image[..., start_y:end_y, start_x:end_x]

                    # Pad image
                    image = functional.pad(
                        image,
                        (pad_left, pad_top, pad_right, pad_bottom),
                        padding_mode=padding_mode,
                        fill=padding_value)

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def add_noise(self, images_arr, do_add_noise, noise_type, noise_spread):
        '''
        Add noise to images

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_add_noise : bool
                N booleans to determine if noise will be added
            noise_type : str
                gaussian, uniform
            noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        for i, images in enumerate(images_arr):
            shape = images.shape
            device = images.device

            if noise_type == 'gaussian':
                images = images + noise_spread * torch.randn(*shape, device=device)
            elif noise_type == 'uniform':
                images = images + noise_spread * (torch.rand(*shape, device=device) - 0.5)
            else:
                raise ValueError('Unsupported noise type: {}'.format(noise_type))

            images_arr[i] = images

        return images_arr

    def adjust_intensity(self, images_arr, do_adjust_intensity, factors):
        '''
        Adjust brightness on each sample
        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_adjust_intensity : bool
                N booleans to determine if intensity is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_adjust_intensity[b]:
                    images[b, ...] = functional.adjust_brightness(image, factors[b])

            images_arr[i] = images

        return images_arr
