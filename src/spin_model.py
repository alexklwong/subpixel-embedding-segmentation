import torch, torchvision
import losses
import networks, log_utils
import global_constants as settings


class SPiNModel(object):
    '''
    Subpixel network (SPiN) that embeds the input into 2x spatial dimension feature maps to
    guide (subpixel guidance, SPG) the output an encoder-decoder architecture

    Arg(s):
        input_channels : int
            number of channels in the input to the subpixel embedding, segmentation encoder decoder
        encoder_type_subpixel_embedding : list[str]
            encoder type for subpixel embedding
        n_filters_encoder_subpixel_embedding : int
            number of filters to use in each block of the subpixel embedding encoder
            e.g. resnet5_subpixel_embedding : [16, 16, 16]
        decoder_type_subpixel_embedding : list[str]
            decoder type for subpixel embedding
        output_channels_subpixel_embedding : int
            number of channels to produce in subpixel embedding
        n_filter_decoder_subpixel_embedding : int
            number of filters to use in each block of decoder
        output_func_subpixel_embedding : str
            output function of subpixel embedding
        encoder_type_segmentation : list[str]
            type of segmentation encoder
        n_filters_encoder_segmentation : list[int]
            list of n_filters for the encoder ateach resolution level
        resolutions_subpixel_guidance : list[int]
            list of resolutions to perform subpixel guidance using space to depth
        n_filters_subpixel_guidance : list [int]
            number of output filters for each space to depth module in subpixel guidance
        n_convolutions_subpixel_guidance : list[int]
            number of convolutions for each subpixel guidance module
        decoder_type_segmentation : list[str]
            type of segmentation decoder
        n_filters_decoder_segmentation : list[str]
            list of n_filters for the decoder at each resolution level
        n_filters_learnable_downsampler : list[int]
            list of filters for each layer in learnable downsampler
        kernel_sizes_learnable_downsampler : list[int]
            list of kernel sizes for each layer in learnable downsampler
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        use_batch_norm : bool
            if set, then applied batch normalization
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 input_channels=settings.N_CHUNK,
                 encoder_type_subpixel_embedding=settings.ENCODER_TYPE_SUBPIXEL_EMBEDDING,
                 n_filters_encoder_subpixel_embedding=settings.N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING,
                 decoder_type_subpixel_embedding=settings.DECODER_TYPE_SUBPIXEL_EMBEDDING,
                 output_channels_subpixel_embedding=settings.OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING,
                 n_filter_decoder_subpixel_embedding=settings.N_FILTER_DECODER_SUBPIXEL_EMBEDDING,
                 output_func_subpixel_embedding=settings.OUTPUT_FUNC,
                 encoder_type_segmentation=settings.ENCODER_TYPE_SEGMENTATION,
                 n_filters_encoder_segmentation=settings.N_FILTERS_ENCODER_SEGMENTATION,
                 resolutions_subpixel_guidance=settings.RESOLUTIONS_SUBPIXEL_GUIDANCE,
                 n_filters_subpixel_guidance=settings.N_FILTERS_SUBPIXEL_GUIDANCE,
                 n_convolutions_subpixel_guidance=settings.N_CONVOLUTIONS_SUBPIXEL_GUIDANCE,
                 decoder_type_segmentation=settings.DECODER_TYPE_SEGMENTATION,
                 n_filters_decoder_segmentation=settings.N_FILTERS_DECODER_SEGMENTATION,
                 n_filters_learnable_downsampler=settings.N_FILTERS_LEARNABLE_DOWNSAMPLER,
                 kernel_sizes_learnable_downsampler=settings.KERNEL_SIZES_LEARNABLE_DOWNSAMPLER,
                 weight_initializer=settings.WEIGHT_INITIALIZER,
                 activation_func=settings.ACTIVATION_FUNC,
                 use_batch_norm=settings.USE_BATCH_NORM,
                 device=torch.device(settings.CUDA)):

        self.encoder_type_subpixel_embedding = encoder_type_subpixel_embedding
        self.decoder_type_subpixel_embedding = decoder_type_subpixel_embedding
        self.encoder_type_segmentation = encoder_type_segmentation
        self.decoder_type_segmentation = decoder_type_segmentation
        self.resolutions_subpixel_guidance = resolutions_subpixel_guidance
        self.device = device

        assert len(resolutions_subpixel_guidance) < len(n_filters_decoder_segmentation)

        '''
        Build subpixel embedding, can be replaced with hand-crafted upsampling (bilinear, nearest)
        '''
        self.use_interpolated_upsampling = \
            'upsample' in encoder_type_subpixel_embedding or \
            'upsample' in decoder_type_subpixel_embedding

        self.use_bilinear_upsampling = \
            self.use_interpolated_upsampling and \
            'bilinear' in self.encoder_type_subpixel_embedding and \
            'bilinear' in self.decoder_type_subpixel_embedding

        self.use_nearest_upsampling = \
            self.use_interpolated_upsampling and \
            'nearest' in self.encoder_type_subpixel_embedding and \
            'nearest' in self.decoder_type_subpixel_embedding

        self.use_subpixel_guidance = \
            'none' not in encoder_type_subpixel_embedding and \
            'none' not in decoder_type_subpixel_embedding and \
            not self.use_interpolated_upsampling

        # Select subpixel embedding encoder
        if 'resnet5_subpixel_embedding' in encoder_type_subpixel_embedding:
            self.encoder_subpixel_embedding = networks.SubpixelEmbeddingEncoder(
                n_layer=5,
                input_channels=input_channels,
                n_filters=n_filters_encoder_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet7_subpixel_embedding' in encoder_type_subpixel_embedding:
            self.encoder_subpixel_embedding = networks.SubpixelEmbeddingEncoder(
                n_layer=7,
                input_channels=input_channels,
                n_filters=n_filters_encoder_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'none' in encoder_type_subpixel_embedding or self.use_interpolated_upsampling:
            self.encoder_subpixel_embedding = None
        else:
            raise ValueError('Unsupported encoder type {}'.format(encoder_type_subpixel_embedding))

        # Latent channels is number channels in the last layer of encoder
        latent_channels_subpixel_embedding = n_filters_encoder_subpixel_embedding[-1]

        # Select subpixel embedding decoder
        if 'subpixel' in decoder_type_subpixel_embedding:
            self.decoder_subpixel_embedding = networks.SubpixelEmbeddingDecoder(
                input_channels=latent_channels_subpixel_embedding,
                output_channels=output_channels_subpixel_embedding,
                scale=2,
                n_filter=n_filter_decoder_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func=output_func_subpixel_embedding,
                use_batch_norm=use_batch_norm)
        elif 'none' in decoder_type_subpixel_embedding or self.use_interpolated_upsampling:
            self.decoder_subpixel_embedding = None
        else:
            raise ValueError('Unsupported decoder type: {}'.format(decoder_type_subpixel_embedding))

        '''
        Build segmentation network
        '''
        # Select segmentation encoder
        if 'resnet18' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=18,
                input_channels=input_channels,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet34' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=34,
                input_channels=input_channels,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet50' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=50,
                input_channels=input_channels,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Unsupported segmentation encoder type: {}'.format(
                encoder_type_segmentation))

        # Build skip connections based on output channels in segmentation encoder
        # n_filters_encoder_segmentation up to last one (omit latent)
        # the prepended 0's are to account for original and 2x resolution skips
        skip_channels_segmentation = [0, 0] + n_filters_encoder_segmentation[:-1]

        if self.use_interpolated_upsampling:
            skip_channels_segmentation[0] = 1

        latent_channels_segmentation = n_filters_encoder_segmentation[-1]

        '''
        Build Subpixel Guidance Modules

        0 -> scale = 1 -> 2x resolution (output size of SubpixelGuidanceDecoder)
        1 -> scale = 2-> 1x resolution
        '''
        if self.use_subpixel_guidance:

            for resolution in resolutions_subpixel_guidance:
                assert resolution <= 5, 'Unsupported resolution for subpixel guidance: {}'

            # Ensure n_filters_subpixel_guidance is compatible with SPG module.
            assert isinstance(resolutions_subpixel_guidance, list) and \
                isinstance(n_filters_subpixel_guidance, list) and \
                isinstance(n_convolutions_subpixel_guidance, list), \
                'Arguments (resolutions, n_filters, n_convolutions) for subpixel guidance must be lists'

            assert len(resolutions_subpixel_guidance) == len(n_filters_subpixel_guidance) and \
                len(resolutions_subpixel_guidance) == len(n_convolutions_subpixel_guidance), \
                'Arguments (resolutions, n_filters, n_convolutions) must have same length'

            scales = [
                int(2 ** s) for s in resolutions_subpixel_guidance
            ]

            self.subpixel_guidance = networks.SubpixelGuidance(
                scales=scales,
                n_filters=n_filters_subpixel_guidance,
                n_convolutions=n_convolutions_subpixel_guidance,
                subpixel_embedding_channels=output_channels_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            # Add skip connections from subpixel embedding starting from the end
            for (idx, n_filters) in zip(resolutions_subpixel_guidance, n_filters_subpixel_guidance):
                # Have a convolution at that resolution
                if n_filters > 0:
                    skip_channels_segmentation[idx] += n_filters
                # No convolution at that resolution -> use # channels from space to depth output
                else:
                    skip_channels_segmentation[idx] += int(4 ** idx * output_channels_subpixel_embedding)

        # If we don't plan on building subpixel guidance
        else:
            self.subpixel_guidance = None

        # Reverse list to build layers for decoder
        skip_channels_segmentation = skip_channels_segmentation[::-1]

        # Segmentation Decoder
        if 'subpixel_guidance' in decoder_type_segmentation:

            if 'learnable_downsampler' not in decoder_type_segmentation:
                n_filters_learnable_downsampler = []
                kernel_sizes_learnable_downsampler = []

            self.decoder_segmentation = networks.SubpixelGuidanceDecoder(
                input_channels=latent_channels_segmentation,
                output_channels=1,
                n_filters=n_filters_decoder_segmentation,
                n_skips=skip_channels_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm=use_batch_norm,
                n_filters_learnable_downsampler=n_filters_learnable_downsampler,
                kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler)
        elif 'generic' in decoder_type_segmentation:
            self.decoder_segmentation = networks.GenericDecoder(
                input_channels=latent_channels_segmentation,
                output_channels=1,
                n_filters=n_filters_decoder_segmentation,
                n_skips=skip_channels_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm=use_batch_norm,
                full_resolution_output='1x' in decoder_type_segmentation)
        else:
            raise ValueError('Unsupported segmentation decoder type: {}'.format(
                decoder_type_segmentation))

        # Ensure that modules are removed if not using subpixel guidance
        if not self.use_subpixel_guidance:
            assert self.encoder_subpixel_embedding is None and \
                self.decoder_subpixel_embedding is None and \
                self.subpixel_guidance is None, \
                'Subpixel encoder and decoder types must be none if not using subpixel guidance:\n' + \
                'encoder_subpixel_embedding={}\n' + \
                'decoder_subpixel_embedding={}'.format(
                    encoder_type_subpixel_embedding, decoder_type_subpixel_embedding)

        # Move to device
        self.to(self.device)

    def forward(self, input_scan):
        '''
        Forwards the input through the network

        Arg(s):
            input_scan : tensor[float32]
                input MRI scan

        Returns:
            list[tensor] : lesion segmentation in a list
        '''

        # Remove extra dimension (should be 1) from N x C x D x H x W to get N x C x H x W
        input_scan = input_scan[:, :, 0, :, :]
        self.input_scan = input_scan

        if self.use_subpixel_guidance:
            # Forward through subpixel embedding to get N x M x 2H x 2W
            latent_subpixel_embedding, \
                skips_subpixel_embedding = self.encoder_subpixel_embedding(input_scan)

            output_subpixel_embedding = \
                self.decoder_subpixel_embedding(latent_subpixel_embedding)
            self.output_subpixel_embedding = output_subpixel_embedding[-1]

        # Forward original tensor through segmentation encoder
        latent_segmentation, skips_segmentation = self.encoder_segmentation(input_scan)

        # Add placeholder for 2x and 1x resolution skips
        skips_segmentation = [None, None] + skips_segmentation

        # Use upsampling instead of subpixel guidance
        if self.use_interpolated_upsampling:

            if self.use_bilinear_upsampling:
                interpolated_scan = torch.nn.functional.interpolate(
                    input_scan,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)
            elif self.use_nearest_upsampling:
                interpolated_scan = torch.nn.functional.interpolate(
                    input_scan,
                    scale_factor=2,
                    mode='nearest')
            else:
                raise ValueError('Must specify bilinear or nearest interpolation type.')
            skips_segmentation[0] = interpolated_scan

        '''
        Forward through Subpixel Guidance
        '''
        if self.use_subpixel_guidance:
            # Calculate desired output shapes for each skip connection in subpixel guidance
            skips_segmentation_shapes = [
                skip.shape[-2:] for skip in skips_segmentation if skip is not None
            ]
            skips_segmentation_shapes = \
                [self.output_subpixel_embedding.shape[-2:], input_scan.shape[-2:]] + skips_segmentation_shapes
            skips_segmentation_shapes = [
                skips_segmentation_shapes[resolution] for resolution in self.resolutions_subpixel_guidance
            ]

            # Space to Depth to build skip connections
            skips_subpixel_guidance = self.subpixel_guidance(
                self.output_subpixel_embedding,
                skips_segmentation_shapes)

            # Concatenate segmentation encoder skips with remaining subpixel guidance skips
            for (resolution, skip_subpixel_guidance) in zip(self.resolutions_subpixel_guidance, skips_subpixel_guidance):
                if skips_segmentation[resolution] is None:
                    skips_segmentation[resolution] = skip_subpixel_guidance
                else:
                    skips_segmentation[resolution] = torch.cat(
                        [skips_segmentation[resolution], skip_subpixel_guidance],
                        dim=1)

        # Forward through decoder & return output
        logits = self.decoder_segmentation(latent_segmentation, skips_segmentation)

        return logits

    def compute_loss(self,
                     output_logits,
                     ground_truth,
                     loss_func_segmentation=settings.LOSS_FUNC_SEGMENTATION,
                     w_cross_entropy=settings.W_CROSS_ENTROPY,
                     w_positive_class=settings.W_POSITIVE_CLASS):
        '''
        Computes the loss function

        Arg(s):
            output_logits : list[tensor[float32]]
                output segmentation logits
            ground_truth : tensor[float32]
                ground-truth segmentation class map
            loss_func_segmentation : list[str]
                list of loss functions to use for super resolution
            w_cross_entropy : float
                weight of cross_entropy loss function
            w_positive_class : float
                weight of positive class penalty
        Returns:
            tensor[float32] : loss (scalar)
        '''

        output_height, output_width = output_logits[-1].shape[-2:]
        target_height, target_width = ground_truth.shape[-2:]

        if output_height > target_height and output_width > target_width:
            height, width = output_height, output_width
            ground_truth = ground_truth.float()
            ground_truth = torch.nn.functional.interpolate(
                ground_truth,
                size=(height, width),
                mode='nearest')
            ground_truth = ground_truth.long()
        else:
            height, width = target_height, target_width

        self.output_logits = output_logits[-1].squeeze()

        # Squeeze ground truth to N x H x W
        self.ground_truth = ground_truth.squeeze(1)

        self.loss_segmentation = 0.0

        # Cross Entropy (implemented with BCE)
        if 'cross_entropy' in loss_func_segmentation:
            self.loss_segmentation = losses.binary_cross_entropy_loss_func(
                src=self.output_logits,
                tgt=self.ground_truth,
                w=torch.tensor(w_positive_class, device=self.device))

        # Dice loss
        elif 'dice' in loss_func_segmentation:
            target = self.ground_truth.to(torch.float32)

            # Take sigmoid
            output_sigmoid = torch.sigmoid(self.output_logits).to(torch.float32)
            self.loss_segmentation = losses.soft_dice_loss_func(
                src=output_sigmoid,
                tgt=target,
                smoothing=1.0)

        return self.loss_segmentation

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        if self.use_subpixel_guidance:
            parameters_subpixel_embedding = \
                list(self.encoder_subpixel_embedding.parameters()) + \
                list(self.decoder_subpixel_embedding.parameters())
        else:
            parameters_subpixel_embedding = []

        parameters_segmentation = \
            list(self.encoder_segmentation.parameters()) + \
            list(self.decoder_segmentation.parameters())

        if self.use_subpixel_guidance:
            parameters_segmentation = \
                parameters_segmentation + \
                list(self.subpixel_guidance.parameters())

        parameters = parameters_subpixel_embedding + parameters_segmentation

        return parameters, parameters_subpixel_embedding, parameters_segmentation

    def train(self):
        '''
        Sets model to training mode
        '''

        if self.use_subpixel_guidance:
            self.encoder_subpixel_embedding.train()
            self.decoder_subpixel_embedding.train()
            self.subpixel_guidance.train()

        self.encoder_segmentation.train()
        self.decoder_segmentation.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        if self.use_subpixel_guidance:
            self.encoder_subpixel_embedding.eval()
            self.decoder_subpixel_embedding.eval()
            self.subpixel_guidance.eval()

        self.encoder_segmentation.eval()
        self.decoder_segmentation.eval()

    def to(self, device):
        '''
        Moves model to device

        Arg(s):
            device : Torch device
                CPU or GPU/CUDA device
        '''

        # Move to device
        if self.use_subpixel_guidance:
            self.encoder_subpixel_embedding.to(device)
            self.decoder_subpixel_embedding.to(device)
            self.subpixel_guidance.to(device)

        self.encoder_segmentation.to(device)
        self.decoder_segmentation.to(device)

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {}

        # Save training state
        checkpoint['train_step'] = step
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save super resolution encoder and decoder weights
        if self.use_subpixel_guidance:
            checkpoint['encoder_subpixel_embedding_state_dict'] = self.encoder_subpixel_embedding.state_dict()
            checkpoint['decoder_subpixel_embedding_state_dict'] = self.decoder_subpixel_embedding.state_dict()
            checkpoint['subpixel_guidance_state_dict'] = self.subpixel_guidance.state_dict()

        checkpoint['encoder_segmentation_state_dict'] = self.encoder_segmentation.state_dict()
        checkpoint['decoder_segmentation_state_dict'] = self.decoder_segmentation.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore encoder and decoder weights
        if self.use_subpixel_guidance:
            self.encoder_subpixel_embedding.load_state_dict(checkpoint['encoder_subpixel_embedding_state_dict'])
            self.decoder_subpixel_embedding.load_state_dict(checkpoint['decoder_subpixel_embedding_state_dict'])
            self.subpixel_guidance.load_state_dict(checkpoint['subpixel_guidance_state_dict'])

        self.encoder_segmentation.load_state_dict(checkpoint['encoder_segmentation_state_dict'])
        self.decoder_segmentation.load_state_dict(checkpoint['decoder_segmentation_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def log_summary(self,
                    input_scan,
                    output_logits,
                    ground_truth,
                    scalar_dictionary,
                    summary_writer,
                    step,
                    n_display=1):
        '''
        Log the input scan, output logits, ground truth, error, and scalars to tensorboard

        Arg(s):
            input_scan : tensor[float32]
                input N x C x 1 x H x W MRI scan
            output_logits : tensor[float32]
                output of segmentation network
            ground_truth : tensor[float32]
                ground truth annotation
            scalar_dictionary : dict[str, float]
                dictionary of scalar name and value to graph
            summary_writer : SummaryWriter
                Tensorboard summary writer
            step : int
                current step in training
            n_display : int
                number of samples from batch to display
        Returns:
            None
        '''

        with torch.no_grad():

            # N x C x H x W case (ensure n_display is between batch size and 1)
            if len(output_logits.shape) == 4:
                n_display = min(output_logits.shape[0], max(n_display, 1))

            if len(input_scan.shape) == 5:
                _, n_chunk, _, o_height, o_width = input_scan.shape
            else:
                _, n_chunk, o_height, o_width = input_scan.shape

            if n_chunk > 1:
                # Shape: N x C x D x H x W
                if len(input_scan.shape) == 5:
                    input_scan = torch.unsqueeze(
                        input_scan[0:n_display, n_chunk // 2, 0, :, :], dim=1)
                # Shape: N x C x H x W or N x D x H x W or N x (C x D) x H x W
                else:
                    input_scan = torch.unsqueeze(
                        input_scan[0:n_display, n_chunk // 2, :, :], dim=1)
            else:
                # Shape: N x 1 x D x H x W
                if len(input_scan.shape) == 5:
                    input_scan = input_scan[0:n_display, :, 0, :, :]
                # Shape: N x 1 x H x W
                else:
                    input_scan = input_scan[0:n_display, ...]

            '''
            Logging segmentation outputs to summary
            '''

            # Make output_segmentation into binary segmentation
            n_height, n_width = output_logits.shape[-2:]

            output_logits = output_logits[0:n_display]
            output_sigmoid = torch.sigmoid(output_logits.to(torch.float32))

            output_segmentation = torch.where(
                output_sigmoid < 0.5,
                torch.zeros_like(output_sigmoid),
                torch.ones_like(output_sigmoid))

            output_segmentation = torch.unsqueeze(output_segmentation, dim=1)

            # Reshape input scan to match output segmentation
            if o_height != n_height or o_width != n_width:
                input_scan = torch.nn.functional.interpolate(
                    input=input_scan,
                    size=(n_height, n_width))

            ground_truth = torch.unsqueeze(ground_truth[0:n_display], dim=1)

            # Resize ground truth if needed to match output segmentation
            ground_truth_height, ground_truth_width = ground_truth.shape[-2:]
            if ground_truth_height != n_height or ground_truth_width != n_width:
                ground_truth = ground_truth.float()
                ground_truth = torch.nn.functional.interpolate(
                    input=ground_truth,
                    size=(n_height, n_width),
                    mode='nearest')
                ground_truth = ground_truth.long()

            error_segmentation = torch.abs(output_segmentation - ground_truth)

            # Log images to summary
            input_scan_summary = log_utils.colorize(input_scan.cpu(), colormap='viridis')
            output_segmentation_summary = log_utils.colorize(255.0 * output_segmentation.cpu(), colormap='gray')
            ground_truth_summary = log_utils.colorize(255.0 * ground_truth.cpu(), colormap='gray')
            error_segmentation_summary = log_utils.colorize(255.0 * error_segmentation.cpu(), colormap='magma')

            display_summary = torch.cat([
                input_scan_summary,
                output_segmentation_summary,
                ground_truth_summary,
                error_segmentation_summary], dim=-1)

            summary_writer.add_image(
                'input_output_groundtruth_error',
                torchvision.utils.make_grid(display_summary, nrow=1),
                global_step=step)

            # Log scalars
            for (metric_name, metric_value) in scalar_dictionary.items():
                summary_writer.add_scalar(metric_name, metric_value, global_step=step)
