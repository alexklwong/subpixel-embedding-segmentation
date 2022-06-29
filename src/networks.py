import net_utils
import torch


'''
Encoder architectures
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetEncoder, self).__init__()

        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        assert len(n_filters) == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        filter_idx = filter_idx + 1

        blocks2 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=False)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=False)

            blocks2.append(block)

        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks3 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=False)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=False)

            blocks3.append(block)

        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks4 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=use_depthwise_separable)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=use_depthwise_separable)

            blocks4.append(block)

        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks5 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=use_depthwise_separable)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=use_depthwise_separable)

            blocks5.append(block)

        self.blocks5 = torch.nn.Sequential(*blocks5)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks6 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):
                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=2,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm,
                        use_depthwise_separable=use_depthwise_separable)
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=1,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm,
                        use_depthwise_separable=use_depthwise_separable)

                blocks6.append(block)

            self.blocks6 = torch.nn.Sequential(*blocks6)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks7 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):
                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=2,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm,
                        use_depthwise_separable=use_depthwise_separable)
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=1,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm,
                        use_depthwise_separable=use_depthwise_separable)

                blocks7.append(block)

            self.blocks7 = torch.nn.Sequential(*blocks7)
        else:
            self.blocks7 = None

    def forward(self, x):
        '''
        Forward input x through a ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]


class SubpixelEmbeddingEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections for subpixel embedding

    Arg(s):
        n_layer : int
            architecture type based on layers: 4, 5, 7, 11
        input_channels : int
            number of channels in input data
        n_filters : list[int]
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[16, 16, 16],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(SubpixelEmbeddingEncoder, self).__init__()

        if n_layer == 5:
            n_blocks = 2
        elif n_layer == 7:
            n_blocks = 3
        elif n_layer == 9:
            n_blocks = 4
        elif n_layer == 11:
            n_blocks = 5
        else:
            raise ValueError('Only supports 5, 7, 9, 11 layer architecture')

        activation_func = net_utils.activation_func(activation_func)

        # First convolution (feature extraction) will be a residual connection
        in_channels, out_channels = [input_channels, n_filters[0]]

        self.conv0 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Mapping to higher dimensional space
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        blocks = []
        for n in range(n_blocks):
            block = net_utils.ResNetBlock(
                in_channels,
                out_channels,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                use_depthwise_separable=False)

            blocks.append(block)
            in_channels, out_channels = [n_filters[1], n_filters[1]]

        self.blocks = torch.nn.Sequential(*blocks)

        in_channels, out_channels = [n_filters[1], n_filters[2]]
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x):
        layers = []

        # First convolution (feature extraction)
        conv0 = self.conv0(x)
        layers.append(conv0)

        # Mapping to higher dimensional space
        blocks = self.blocks(conv0)
        layers.append(blocks)

        # Learn the residual from high dimensional features
        conv1 = self.conv1(blocks)
        layers.append(conv1)

        latent = conv1 + conv0

        return latent, layers


class AtrousResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        atrous_spatial_pyramid_pool_dilations : list[int]
            list of dilation rates for atrous spatial pyramid pool (ASPP)
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
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 atrous_spatial_pyramid_pool_dilations=None,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousResNetEncoder, self).__init__()

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
            atrous_resnet_block = net_utils.AtrousResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
            atrous_resnet_block = net_utils.AtrousResNetBlock
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        assert len(n_filters) == len(n_blocks) + 1

        activation_func = net_utils.activation_func(activation_func)
        dilation = 2
        in_channels, out_channels = [input_channels, n_filters[0]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels, out_channels = [n_filters[0], n_filters[1]]

        blocks2 = []
        for n in range(n_blocks[0]):
            if n == 0:
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                blocks2.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                blocks2.append(block)
        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        blocks3 = []
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        for n in range(n_blocks[1]):
            if n == 0:
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                blocks3.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                blocks3.append(block)
        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 with 2x dilation
        blocks4 = []
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        for n in range(n_blocks[2]):
            if n == 0:
                block = atrous_resnet_block(
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                dilation = dilation * 2
                blocks4.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                blocks4.append(block)
        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/8 with 4x dilation
        blocks5 = []
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        for n in range(n_blocks[3]):
            if n == 0:
                block = atrous_resnet_block(
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                dilation = dilation * 2
                blocks5.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                blocks5.append(block)
        self.blocks5 = torch.nn.Sequential(*blocks5)

        if atrous_spatial_pyramid_pool_dilations is not None:
            self.atrous_spatial_pyramid_pool = net_utils.AtrousSpatialPyramidPooling(
                in_channels,
                out_channels,
                dilations=atrous_spatial_pyramid_pool_dilations,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        else:
            self.atrous_spatial_pyramid_pool = torch.nn.Identity()

    def forward(self, x):
        '''
        Forward input x through an atrous ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 with 2x dilation
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/8 with 4x dilation
        # ASPP only used if dilations are given, otherwise pass through (identity)
        block5 = self.blocks5(layers[-1])
        layers.append(self.atrous_spatial_pyramid_pool(block5))

        return layers[-1], layers[1:-1]


class VGGNetEncoder(torch.nn.Module):
    '''
    VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_layer : int
            architecture type based on layers: 8, 11, 13
        n_filters : list
            number of filters to use for each block
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
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(VGGNetEncoder, self).__init__()

        if n_layer == 8:
            n_convolutions = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convolutions = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convolutions = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')

        for n in range(len(n_filters) - len(n_convolutions) - 1):
            n_convolutions = n_convolutions + [n_convolutions[-1]]

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        assert len(n_filters) == len(n_convolutions)

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        stride = 1 if n_convolutions[block_idx] - 1 > 0 else 2
        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        if n_convolutions[block_idx] - 1 > 0:
            self.conv1 = torch.nn.Sequential(
                conv1,
                net_utils.VGGNetBlock(
                    out_channels,
                    out_channels,
                    n_convolution=n_convolutions[filter_idx] - 1,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm,
                    use_depthwise_separable=False))
        else:
            self.conv1 = conv1

        # Resolution 1/2 -> 1/4
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv2 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            use_depthwise_separable=False)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv3 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            use_depthwise_separable=False)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv4 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            use_depthwise_separable=use_depthwise_separable)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv5 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            use_depthwise_separable=use_depthwise_separable)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.conv6 = net_utils.VGGNetBlock(
                in_channels,
                out_channels,
                n_convolution=n_convolutions[block_idx],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                use_depthwise_separable=use_depthwise_separable)
        else:
            self.conv6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.conv7 = net_utils.VGGNetBlock(
                in_channels,
                out_channels,
                n_convolution=n_convolutions[block_idx],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                use_depthwise_separable=use_depthwise_separable)
        else:
            self.conv7 = None

    def forward(self, x):
        '''
        Forward input x through a VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/32
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.conv6 is not None:
            layers.append(self.conv6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.conv7 is not None:
            layers.append(self.conv7(layers[-1]))

        return layers[-1], layers[1:-1]


class AtrousVGGNetEncoder(torch.nn.Module):
    '''
    Atrous VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_layer : int
            architecture type based on layers: 8, 11, 13
        n_filters : list
            number of filters to use for each block
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
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousVGGNetEncoder, self).__init__()

        if n_layer == 8:
            n_convolutions = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convolutions = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convolutions = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')

        assert len(n_filters) == len(n_convolutions)

        activation_func = net_utils.activation_func(activation_func)
        dilation = 2

        # Resolution 1/1 -> 1/2
        stride = 1 if n_convolutions[0] - 1 > 0 else 2
        in_channels, out_channels = [input_channels, n_filters[0]]

        conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        if n_convolutions[0] - 1 > 0:
            self.conv1 = torch.nn.Sequential(
                conv1,
                net_utils.VGGNetBlock(
                    out_channels,
                    out_channels,
                    n_convolution=n_convolutions[0] - 1,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm))
        else:
            self.conv1 = conv1

        # Resolution 1/2 -> 1/4
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        self.conv2 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[1],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/4 -> 1/8
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        self.conv3 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[2],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/8 with 2x dilation
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        self.conv4 = net_utils.AtrousVGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[3],
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/8 with 4x dilation
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        self.conv5 = net_utils.AtrousVGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[4],
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x):
        '''
        Forward input x through an atrous VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 with 2x dilation
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/8 with 4x dilation
        layers.append(self.conv5(layers[-1]))

        return layers[-1], layers[1:-1]


'''
Decoder architectures
'''
class GenericDecoder(torch.nn.Module):
    '''
    Generic decoder

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_filters : list[int]
            number of filters to use at each decoder block
        n_skips : list[int]
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
        output_func : str
            activation function for output
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types available: transpose, up
        full_resolution_output : bool
            if set then output at full resolution instead of 1/2
    '''
    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_filters=[256, 128, 64, 32, 16, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up',
                 full_resolution_output=False):
        super(GenericDecoder, self).__init__()

        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Resolution 1/32 -> 1/16
        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[0], n_filters[0]
        ]
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        in_channels, skip_channels, out_channels = [
            n_filters[0], n_skips[1], n_filters[1]
        ]
        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/8 -> 1/4
        in_channels, skip_channels, out_channels = [
            n_filters[1], n_skips[2], n_filters[2]
        ]
        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/4 -> 1/2
        in_channels, skip_channels, out_channels = [
            n_filters[2], n_skips[3], n_filters[3]
        ]
        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        if full_resolution_output:
            in_channels, skip_channels, out_channels = [
                n_filters[3], 0, n_filters[4]
            ]
            self.deconv0 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)
        else:
            self.deconv0 = None

        # Output layer
        self.output = net_utils.Conv2d(
            in_channels=out_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x, skips):
        '''
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Resolution 1/32 -> 1/16
        n = len(skips) - 1
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n - 1
        layers.append(self.deconv3(layers[-1], skips[n]))

        # Resolution 1/8 -> 1/4
        n = n - 1
        layers.append(self.deconv2(layers[-1], skips[n]))

        # Resolution 1/4 -> 1/2
        n = n - 1
        layers.append(self.deconv1(layers[-1], skips[n]))

        # Resolution 1/2 -> 1
        n = n - 1
        if self.deconv0 is not None:
            layers.append(self.deconv0(layers[-1], None))

        # Output (1 x H x W)
        outputs.append(self.output(layers[-1]))

        return outputs


class MultiScaleDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : list[int]
            number of filters to use at each decoder block
        n_skips : list[int]
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=4,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='transpose'):
        super(MultiScaleDecoder, self).__init__()

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2

        filter_idx = 0

        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[filter_idx], n_filters[filter_idx]
        ]

        # Resolution 1/128 -> 1/64
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv6 = None

        # Resolution 1/64 -> 1/32
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv5 = None

        # Resolution 1/32 -> 1/16
        if network_depth > 4:
            self.deconv4 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv4 = None

        # Resolution 1/16 -> 1/8
        if network_depth > 3:
            self.deconv3 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            if self.n_resolution > 3:
                self.output3 = net_utils.Conv2d(
                    out_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=None,
                    use_batch_norm=False,
                    use_instance_norm=False)
            else:
                self.output3 = None

            # Resolution 1/8 -> 1/4
            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]

            if self.n_resolution > 3:
                skip_channels = skip_channels + output_channels
        else:
            self.deconv3 = None

        if network_depth > 2:
            self.deconv2 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            if self.n_resolution > 2:
                self.output2 = net_utils.Conv2d(
                    out_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=output_func,
                    use_batch_norm=False,
                    use_instance_norm=False)
            else:
                self.output2 = None

            # Resolution 1/4 -> 1/2
            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]

            if self.n_resolution > 2:
                skip_channels = skip_channels + output_channels
        else:
            self.deconv2 = None

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 1:
            self.output1 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False,
                use_instance_norm=False)
        else:
            self.output1 = None

        # Resolution 1/2 -> 1/1
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        self.output0 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x, skips, shape=None):
        '''
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
            shape : tuple[int]
                (height, width) tuple denoting output size
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Start at the end and walk backwards through skip connections
        n = len(skips) - 1

        # Resolution 1/128 -> 1/64
        if self.deconv6 is not None:
            layers.append(self.deconv6(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/64 -> 1/32
        if self.deconv5 is not None:
            layers.append(self.deconv5(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/32 -> 1/16
        if self.deconv4 is not None:
            layers.append(self.deconv4(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/16 -> 1/8
        if self.deconv3 is not None:
            layers.append(self.deconv3(layers[-1], skips[n]))

            if self.n_resolution > 3:
                output3 = self.output3(layers[-1])
                outputs.append(output3)

                if n > 0:
                    upsample_output3 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        size=skips[n-1].shape[-2:],
                        mode='bilinear',
                        align_corners=True)
                else:
                    upsample_output3 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)

            n = n - 1

        # Resolution 1/8 -> 1/4
        if self.deconv2 is not None:
            if skips[n] is not None:
                skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
            else:
                skip = skips[n]
            layers.append(self.deconv2(layers[-1], skip))

            if self.n_resolution > 2:
                output2 = self.output2(layers[-1])
                outputs.append(output2)

                if n > 0:
                    upsample_output2 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        size=skips[n-1].shape[-2:],
                        mode='bilinear',
                        align_corners=True)
                else:
                    upsample_output2 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)

            n = n - 1

        # Resolution 1/4 -> 1/2
        if skips[n] is not None:
            skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        else:
            skip = skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)

            if n > 0:
                upsample_output1 = torch.nn.functional.interpolate(
                    input=outputs[-1],
                    size=skips[n-1].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
            else:
                upsample_output1 = torch.nn.functional.interpolate(
                    input=outputs[-1],
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                # If there is skip connection at layer 0
                if skips[n] is not None and n == 0:
                    skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                else:
                    skip = upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            else:

                if skips[n] is not None and n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1], shape=shape[-2:]))

            output0 = self.output0(layers[-1])

        outputs.append(output0)

        return outputs


class SubpixelGuidanceDecoder(torch.nn.Module):
    '''
    Subpixel guidance (SPG) decoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_filters : list[int]
            number of filters to use at each decoder block
        n_skips : list[int]
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
        output_func : str
            activation function for output
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types available: transpose, up
        n_filters_learnable_downsampler : list[int]
            list of filters for each layer in learnable downsampler
        kernel_sizes_learnable_downsampler : list[int]
            list of kernel sizes for each layer in learnable downsampler
    '''
    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_filters=[256, 128, 64, 32, 16, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up',
                 n_filters_learnable_downsampler=[16, 16],
                 kernel_sizes_learnable_downsampler=[3, 3]):
        super(SubpixelGuidanceDecoder, self).__init__()

        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Resolution 1/32 -> 1/16
        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[0], n_filters[0]
        ]
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        in_channels, skip_channels, out_channels = [
            n_filters[0], n_skips[1], n_filters[1]
        ]
        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/8 -> 1/4
        in_channels, skip_channels, out_channels = [
            n_filters[1], n_skips[2], n_filters[2]
        ]
        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/4 -> 1/2
        in_channels, skip_channels, out_channels = [
            n_filters[2], n_skips[3], n_filters[3]
        ]
        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1/2 -> 1
        in_channels, skip_channels, out_channels = [
            n_filters[3], n_skips[4], n_filters[4]
        ]
        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Resolution 1 -> 2
        in_channels, skip_channels, out_channels = [
            n_filters[4], n_skips[5], n_filters[5]
        ]
        self.deconv_super_resolution = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        # Output layer
        self.output = net_utils.Conv2d(
            in_channels=out_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False,
            use_instance_norm=False)

        if len(n_filters_learnable_downsampler) > 0:
            self.learnable_downsampler = LearnableDownsampler(
                latent_channels=out_channels,
                output_channels=output_channels,
                n_filters=n_filters_learnable_downsampler,
                kernel_sizes=kernel_sizes_learnable_downsampler,
                scale=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func)
        else:
            self.learnable_downsampler = None

    def forward(self, x, skips):
        '''
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Resolution 1/32 -> 1/16
        n = len(skips) - 1
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n - 1
        layers.append(self.deconv3(layers[-1], skips[n]))

        # Resolution 1/8 -> 1/4
        n = n - 1
        layers.append(self.deconv2(layers[-1], skips[n]))

        # Resolution 1/4 -> 1/2
        n = n - 1
        layers.append(self.deconv1(layers[-1], skips[n]))

        # Resolution 1/2 -> 1
        n = n - 1
        if skips[n] is None:
            layers.append(self.deconv0(layers[-1]))
        else:
            layers.append(self.deconv0(layers[-1], skips[n]))

        # Resolution 1 -> 2
        n = n - 1
        if skips[n] is None:
            layers.append(self.deconv_super_resolution(layers[-1]))
        else:
            layers.append(self.deconv_super_resolution(layers[-1], skips[n]))

        # Output (1 x 2H x 2W)
        outputs.append(self.output(layers[-1]))

        if self.learnable_downsampler is not None:
            # Output (1 x H x W)
            outputs.append(self.learnable_downsampler(layers[-1], outputs[-1]))

        return outputs


class SubpixelEmbeddingDecoder(torch.nn.Module):
    '''
    Subpixel embedding decoder

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        scale : int
            scale multiplier to super resolve input
        n_filter : int
            number of filters to use
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
        output_func : str
            activation function for output
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 scale=2,
                 n_filter=16,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(SubpixelEmbeddingDecoder, self).__init__()

        assert scale % 2 == 0

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        in_channels = input_channels

        blocks = []
        for n in range(scale // 2):
            # out_channels is 4 times the number of filters for each 2x upsample
            out_channels = 4 * n_filter

            conv = net_utils.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)

            subpixel = net_utils.DepthToSpace(scale=2)

            # in_channels are now n_filters after subpixel layer
            in_channels = n_filter

            # Append convolution and subpixel layer to sequential
            blocks.append(conv)
            blocks.append(subpixel)

        self.upsampling_blocks = torch.nn.Sequential(*blocks)

        self.output = net_utils.Conv2d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward latent vector x through subpixel decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
        Returns:
            list[torch.Tensor[float32]] : list of outputs
        '''

        outputs = []

        upsampling_blocks = self.upsampling_blocks(x)
        outputs.append(self.output(upsampling_blocks))

        return outputs


class SubpixelGuidance(torch.nn.Module):
    '''
    Perform space to depth for varying output sizes

    Arg(s):
        resolutions : list[int]
            List of resolutions to perform space to depth
        n_filters : list[int]
            number of convolutions for each space to depth output to undergo
        n_convolutions : list[int]
            number of convolutions for each space to depth output to undergo
        subpixel_embedding_channels : int
            number of channels for the output of subpixel embedding network
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''
    def __init__(self,
                 scales,
                 n_filters,
                 n_convolutions,
                 subpixel_embedding_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(SubpixelGuidance, self).__init__()

        self.scales = scales
        self.s2d_modules = torch.nn.ModuleList()
        self.conv_modules = torch.nn.ModuleList()

        activation_func = net_utils.activation_func(activation_func)

        for (scale, n_filter, n_convolution) in zip(scales, n_filters, n_convolutions):

            # Create module list of S2D modules
            if scale == 1:
                self.s2d_modules.append(torch.nn.Identity())
            else:
                self.s2d_modules.append(net_utils.SpaceToDepth(scale=scale))

            # Create module list of conv blocks
            if n_filter > 0:
                in_channels = subpixel_embedding_channels * scale ** 2
                out_channels = n_filter

                conv_block = []

                # Use a minimum of 1 convolution
                n_convolution = max(1, n_convolution)

                for n in range(n_convolution):
                    conv_block.append(net_utils.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm))

                    in_channels = out_channels

                self.conv_modules.append(torch.nn.Sequential(*conv_block))
            else:
                self.conv_modules.append(torch.nn.Identity())

    def forward(self, x, output_shapes):
        '''
        Forward latent vector x through subpixel guidance network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            output_shapes : list[list[int, int]]
                shape of output tensor
        Returns:
            list[torch.Tensor[float32]] : list of outputs
        '''

        # Check that length of output_shapes matches self.scales
        assert len(output_shapes) == len(self.scales)

        outputs = []

        # Iterate through S2D modules
        for (s2d_module, conv_module, output_shape) in zip(self.s2d_modules, self.conv_modules, output_shapes):
            if len(output_shape) != 2:
                raise ValueError('Expected output shape to be size 2, received {}.'.format(len(output_shape)))

            # Perform Space to Depth
            output = s2d_module(x)

            if output.shape != output_shape:
                output = torch.nn.functional.interpolate(
                    output,
                    size=output_shape,
                    mode='nearest')

            output = conv_module(output)

            outputs.append(output)

        return outputs

class LearnableDownsampler(torch.nn.Module):
    '''
    Downsample using learnable linear combination of local features

    Arg(s):
        latent_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels in output
        n_filter : int
            number of filters to use
        scale : int
            scale to downsample
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
    '''
    def __init__(self,
                 latent_channels,
                 output_channels,
                 n_filters,
                 kernel_sizes,
                 scale,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(LearnableDownsampler, self).__init__()

        assert len(n_filters) == len(kernel_sizes)

        self.output_channels = output_channels

        if isinstance(activation_func, str):
            activation_func = net_utils.activation_func(activation_func)

        self.space_to_depth = net_utils.SpaceToDepth(scale=scale)

        in_channels = latent_channels * int(scale ** 2)

        conv_block = []
        for n_filter, kernel_size in zip(n_filters, kernel_sizes):
            conv_block.append(
                net_utils.Conv2d(
                    in_channels=in_channels,
                    out_channels=n_filter,
                    kernel_size=kernel_size,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=False))

            in_channels = n_filter

        self.conv_block = torch.nn.Sequential(*conv_block)

        self.weights = net_utils.Conv2d(
            in_channels=n_filter,
            out_channels=output_channels * int(scale ** 2),
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=net_utils.activation_func('linear'),
            use_batch_norm=False)

    def forward(self, latent, x):
        '''
        Arg(s):
            latent : torch.Tensor[float32]
                N x C x H x W high resolution latent
            x : torch.Tensor[float32]
                N x c x H x W high resolution predictions
        Returns:
            torch.Tensor[float32] : downsampled output
        '''

        # From N x C x H x W to N x (C * S^2) x (H / S) x (W / S)
        latent = self.space_to_depth(latent)
        x = self.space_to_depth(x)

        n_batch, _, n_height, n_width = x.shape

        conv_block = self.conv_block(latent)

        # Weights for linear combination: N x S^2 x (H / S) x (W / S)
        weights = self.weights(conv_block)
        weights = torch.softmax(weights, dim=1)

        # Reshape to N x C x S^2 x (H / S) x (W / S)
        x = x.view(n_batch, self.output_channels, -1, n_height, n_width)

        # Reshape to N x 1 x S^2 x (H / S) x (W / S)
        weights = weights.view(n_batch, 1, -1, n_height, n_width)

        # Multiply and broadcast to yield N x C x S^2 x (H / S) x (W / S), then sum
        output = torch.sum(weights * x, dim=2)

        return output
