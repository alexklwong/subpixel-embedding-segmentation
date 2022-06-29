import torch


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        torch.nn.Module : activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(Conv2d, self).__init__()

        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class DepthwiseSeparableConv2d(torch.nn.Module):
    '''
    Depthwise separable convolution class
    Performs
    1. separate k x k convolution per channel (depth-wise)
    2. 1 x 1 convolution across all channels (point-wise)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(DepthwiseSeparableConv2d, self).__init__()

        padding = kernel_size // 2

        self.conv_depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=in_channels)

        self.conv_pointwise = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv_depthwise.weight)
            torch.nn.init.kaiming_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv_depthwise.weight)
            torch.nn.init.xavier_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv_depthwise.weight)
            torch.nn.init.xavier_uniform_(self.conv_pointwise.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.conv = torch.nn.Sequential(
            self.conv_depthwise,
            self.conv_pointwise)

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

        self.activation_func = activation_func

    def forward(self, x):
        '''
        Forward input x through a depthwise convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class AtrousConv2d(torch.nn.Module):
    '''
    2D atrous convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousConv2d, self).__init__()

        padding = dilation

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through an atrous convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(TransposeConv2d, self).__init__()

        padding = kernel_size // 2

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.deconv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a transposed convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        deconv = self.deconv(x)

        if self.use_batch_norm:
            deconv = self.batch_norm(deconv)
        elif self.use_instance_norm:
            deconv = self.instance_norm(deconv)

        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, shape):
        '''
        Forward input x through an up convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv


class FullyConnected(torch.nn.Module):
    '''
    Fully connected layer

    Arg(s):
        in_channels : int
            number of input neurons
        out_channels : int
            number of output neurons
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        dropout_rate : float
            probability to use dropout
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 dropout_rate=0.00):
        super(FullyConnected, self).__init__()

        self.fully_connected = torch.nn.Linear(in_features, out_features)

        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fully_connected.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        if dropout_rate > 0.00 and dropout_rate <= 1.00:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        '''
        Forward input x through a fully connected block

        Arg(s):
            x : torch.Tensor[float32]
                N x C input tensor
        Returns:
            torch.Tensor[float32] : N x K output tensor
        '''

        fully_connected = self.fully_connected(x)

        if self.activation_func is not None:
            fully_connected = self.activation_func(fully_connected)

        if self.dropout is not None:
            return self.dropout(fully_connected)
        else:
            return fully_connected


class DepthToSpace(torch.nn.Module):
    '''
    Depth to space module

    Arg(s):
        scale : int
            scale to increase space by
        native_impl : bool
            if true, use torch's PixelShuffle
    '''
    def __init__(self, scale=2, native_impl=True):
        super().__init__()

        self.scale = scale
        self.native_impl = native_impl

        if native_impl:
            self.depth_to_space = torch.nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):

        if self.native_impl:
            return self.depth_to_space(x)

        else:
            n_batch, n_channel, n_height, n_width = x.size()

            #  Rearrange to (N, S, S, C // S^2, H, W)
            x = x.view(
                n_batch,
                self.scale,
                self.scale,
                n_channel // (self.scale ** 2),
                n_height,
                n_width)

            # Permute to (N, C // S^2, H, S, W, S)
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()

            # Rearrange to (N, C // S^2, H * S, W * S)
            x = x.view(
                n_batch,
                n_channel // (self.scale ** 2),
                n_height * self.scale,
                n_width * self.scale)

            return x


class SpaceToDepth(torch.nn.Module):
    '''
    Space to depth module

    Arg(s):
        scale : int
            scale to which decrease space by
        native_impl : bool
            if true, use torch's unfold function
    '''
    def __init__(self, scale=2, native_impl=True):
        super().__init__()

        self.scale = scale
        self.native_impl = native_impl

    def forward(self, x):

        if self.native_impl:
            n_batch, n_channel, n_height, n_width = x.size()

            unfolded_x = torch.nn.functional.unfold(x, self.scale, stride=self.scale)

            # Rearrange to (N, C * S^2, H // S, W // S)
            x = unfolded_x.view(
                n_batch,
                n_channel * self.scale ** 2,
                n_height // self.scale,
                n_width // self.scale)

            return x

        else:
            n_batch, n_channel, n_height, n_width = x.size()

            # Rearrange to (N, C, H // S, S, W // S, S)
            x = x.view(
                n_batch,
                n_channel,
                n_height // self.scale,
                self.scale,
                n_width // self.scale,
                self.scale)

            # Permute to (N, S, S, C, H // S, W // S)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()

            # Rearrange (N, C * S^2, H // S, W // S)
            x = x.view(
                n_batch,
                n_channel * (self.scale ** 2),
                n_height // self.scale,
                n_width // self.scale)

            return x


'''
Network encoder blocks
'''
class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv3 = conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ResNet bottleneck block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv3 + X)


class AtrousResNetBlock(torch.nn.Module):
    '''
    Basic atrous ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(AtrousResNetBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = AtrousConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through an atrous ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)

        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(VGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


class AtrousVGGNetBlock(torch.nn.Module):
    '''
    Atrous VGGNet block class
    (last block performs atrous convolution instead of convolution with stride)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        dilation : int
            dilation of atrous convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 dilation=2,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(AtrousVGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = AtrousConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through an atrous VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


class AtrousSpatialPyramidPooling(torch.nn.Module):
    '''
    Atrous Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilations : list[int]
            dilations for different atrous convolution of each branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=[6, 12, 18],
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousSpatialPyramidPooling, self).__init__()

        output_channels = out_channels // (len(dilations) + 1)

        # Point-wise 1 by 1 convolution branch
        self.conv1 = Conv2d(
            in_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # 3 by 3 convolutions with different dilation rate
        self.atrous_convs = torch.nn.ModuleList()

        for dilation in dilations:
            atrous_conv = AtrousConv2d(
                in_channels,
                output_channels,
                kernel_size=3,
                dilation=dilation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            self.atrous_convs.append(atrous_conv)

        # Global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.global_pool_conv = Conv2d(
            in_channels,
            output_channels,
            kernel_size=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Fuse point-wise (1 by 1) convolution, atrous convolutions, global pooling
        self.conv_fuse = Conv2d(
            (len(dilations) + 2) * output_channels,
            out_channels,
            kernel_size=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ASPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # List to hold all branches
        branches = []

        # Point-wise (1 by 1) convolution branch
        branches.append(self.conv1(x))

        # Atrous branches
        for atrous_conv in self.atrous_convs:
            branches.append(atrous_conv(x))

        # Global pooling branch
        global_pool = self.global_pool(x)
        global_pool = self.global_pool_conv(global_pool)
        global_pool = torch.nn.functional.interpolate(
            global_pool,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=True)
        branches.append(global_pool)

        return self.conv_fuse(torch.cat(branches, dim=1))


class SpatialPyramidPooling(torch.nn.Module):
    '''
    Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_sizes : list[int]
            pooling kernel size of each branch
        pool_func : str
            max, average
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 pool_func='max',
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(SpatialPyramidPooling, self).__init__()

        output_channels = out_channels // len(kernel_sizes)

        # List of pooling kernel sizes
        self.kernel_sizes = kernel_sizes

        if pool_func == 'max':
            self.pool_func = torch.nn.functional.max_pool2d
        elif pool_func == 'average':
            self.pool_func = torch.nn.functional.avg_pool2d
        else:
            raise ValueError('Unsupported pooling function: {}'.format(pool_func))

        # List of convolutions to compress feature maps
        self.convs = torch.nn.ModuleList()

        for n in kernel_sizes:
            conv = Conv2d(
                in_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            self.convs.append(conv)

        self.conv_fuse = torch.nn.Sequential(
            Conv2d(
                2 * len(kernel_sizes) * output_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False,
                use_instance_norm=False))

    def forward(self, x):
        '''
        Forward input x through SPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # List to hold all branches
        branches = [x]

        # Pyramid pooling branches
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            pool = self.pool_func(
                x,
                kernel_size=(kernel_size, kernel_size),
                stride=(kernel_size, kernel_size))

            pool = torch.nn.functional.interpolate(
                pool,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True)

            branches.append(conv(pool))

        return self.conv_fuse(torch.cat(branches, dim=1))


'''
Network decoder blocks
'''
'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        deconv_type : str
            deconvolution types: transpose, up
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up',
                 use_depthwise_separable=False):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.deconv_type = deconv_type

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)

        concat_channels = skip_channels + out_channels

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv = conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, skip=None, shape=None):
        '''
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x H x W skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':

            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)
            # print(shape)
        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)
