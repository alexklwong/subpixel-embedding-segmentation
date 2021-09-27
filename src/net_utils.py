import torch


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        func : activation function
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
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(Conv2d, self).__init__()

        self.use_batch_norm = use_batch_norm
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

        self.activation_func = activation_func

        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv = self.conv(x)
        conv = self.batch_norm(conv) if self.use_batch_norm else conv

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
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(TransposeConv2d, self).__init__()

        self.use_batch_norm = use_batch_norm
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
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass

        self.activation_func = activation_func

        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        deconv = self.deconv(x)
        deconv = self.batch_norm(deconv) if self.use_batch_norm else deconv

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
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x, shape):
        upsample = torch.nn.functional.interpolate(x, size=shape)
        conv = self.conv(upsample)
        return conv


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
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

    def forward(self, x):
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
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv3 = Conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

    def forward(self, x):
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
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(VGGNetBlock, self).__init__()

        layers = []
        for n in range(n_convolution - 1):
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connections (U-Net)

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
        deconv_type : str
            deconvolution types: transpose, up
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 deconv_type='transpose'):
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
                use_batch_norm=use_batch_norm)

        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

        concat_channels = skip_channels + out_channels
        self.conv = Conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x, skip=None):

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':
            if skip is not None:
                shape = skip.shape[2:4]
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)

        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        concat = self.conv(concat)

        return concat
