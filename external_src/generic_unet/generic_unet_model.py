import torch, torchvision, sys

sys.path.insert(0, 'src')
import losses
import networks, log_utils
import global_constants as settings

class GenericUNetModel(object):
    '''
    Model to super resolve MRI scan and perform segmentation

    Arg(s):
        input_channels_segmentation :int
            number of channels in the input
        encoder_type_segmentation : list[str]
            type of segmentation encoder
        n_filters_encoder_segmentation : list[int]
            list of n_filters for each resolution level
        decoder_type_segmentation : list[str]
            type of segmentation decoder
        n_filters_decoder_segmentation : list[str]
            list of n_filters for decoder for each resolution level
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
                 input_channels_segmentation=settings.N_CHUNK,
                 encoder_type_segmentation=settings.ENCODER_TYPE_SEGMENTATION,
                 n_filters_encoder_segmentation=settings.N_FILTERS_ENCODER_SEGMENTATION,
                 decoder_type_segmentation=settings.DECODER_TYPE_SEGMENTATION,
                 n_filters_decoder_segmentation=settings.N_FILTERS_DECODER_SEGMENTATION,
                 weight_initializer=settings.WEIGHT_INITIALIZER,
                 activation_func=settings.ACTIVATION_FUNC,
                 use_batch_norm=settings.USE_BATCH_NORM,
                 device=torch.device(settings.CUDA)):

        self.encoder_type_segmentation = encoder_type_segmentation
        self.decoder_type_segmentation = decoder_type_segmentation
        self.device = device

        '''
        Build segmentation network
        '''
        # Select segmentation encoder
        if 'resnet18' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=18,
                input_channels=input_channels_segmentation,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'vggnet13' in encoder_type_segmentation:
            self.encoder_segmentation = networks.VGGNetEncoder(
                n_layer=13,
                input_channels=input_channels_segmentation,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            raise ValueError("Unsupported segmentation encoder type: {}".format(
                encoder_type_segmentation))

        latent_channels_segmentation = n_filters_encoder_segmentation[-1]

        # Build skip connections starting with Segmentation
        # n_filters_encoder_segmentation up to last one (omit latent)
        skip_channels_segmentation = n_filters_encoder_segmentation[:-1]
        # reverse and add 0 for last resolution
        skip_channels_segmentation = \
            skip_channels_segmentation[::-1]

        # Segmentation Decoder
        if 'generic' in decoder_type_segmentation:
            self.decoder_segmentation = networks.GenericDecoder(
                input_channels=latent_channels_segmentation,
                output_channels=1,
                n_filters=n_filters_decoder_segmentation,
                n_skips=skip_channels_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm=use_batch_norm)
        else:
            raise ValueError("Unsupported segmentation decoder type: {}".format(
                decoder_type_segmentation))
        # Move to device
        self.to(self.device)

    def forward(self, input_scan):
        '''
        Forwards the input through the network

        Arg(s):
            input_scan : tensor
                input MRI scan

        Returns:
            list[tensor] : lesion segmentation in a list
        '''

        input_scan = input_scan[:, :, 0, :, :]
        self.input_scan = input_scan

        # Forward original tensor through segmentation encoder
        latent_segmentation, skips_segmentation = self.encoder_segmentation(self.input_scan)

        # Forward through decoder & return output
        outputs = self.decoder_segmentation(latent_segmentation, skips_segmentation)

        return outputs

    def compute_loss(self,
                     output_logits,
                     ground_truth,
                     loss_func_segmentation=settings.LOSS_FUNC_SEGMENTATION,
                     w_cross_entropy=settings.W_CROSS_ENTROPY,
                     w_positive_class=settings.W_POSITIVE_CLASS):
        '''
        Computes the loss function

        Arg(s):
            output_logits : list[tensor]
                output segmentation logits
            ground_truth : tensor
                ground-truth segmentation class map
            loss_func_segmentation : list
                list of loss functions to use for super resolution
            w_cross_entropy : float
                weight of cross_entropy loss function
            w_positive_class : float
                weight of positive class penalty
        Returns:
            tensor : loss (scalar)
        '''

        n_height, n_width = self.input_scan.shape[-2:]

        '''
        Compute loss for segmentation
        '''
        output_height, output_width = output_logits[-1].shape[-2:]
        target_height, target_width = ground_truth.shape[-2:]

        if output_height != target_height and output_width != target_width:
            height, width = target_height, target_width
            output_logits[-1] = torch.nn.functional.interpolate(
                output_logits[-1],
                size=(height, width),
                mode='bilinear',
                align_corners=True
            )
        else:
            height, width = target_height, target_width

        self.output_logits = output_logits[-1].squeeze()

        # Squeeze GT to N x H x W
        self.ground_truth = ground_truth.squeeze(1)

        self.loss_segmentation = 0.0

        # Cross Entropy (implemented with BCE)
        if 'cross_entropy' in loss_func_segmentation:
            self.loss_segmentation = losses.binary_cross_entropy_loss_func(
                src=self.output_logits,
                tgt=self.ground_truth,
                w=torch.tensor([w_positive_class], device=self.device))

        # Dice loss
        elif 'dice' in loss_func_segmentation:
            target = self.ground_truth.to(torch.float32)

            # Take sigmoid
            output_sigmoid = torch.sigmoid(output_logits).to(torch.float32)
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

        parameters = \
            list(self.encoder_segmentation.parameters()) + \
            list(self.decoder_segmentation.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder_segmentation.train()
        self.decoder_segmentation.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder_segmentation.eval()
        self.decoder_segmentation.eval()

    def to(self, device):
        '''
        Moves model to device

        Arg(s):
            device : Torch device
                CPU or GPU/CUDA device
        '''

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
            input_scan : tensor
                input MRI scan
            output_logits : tensor
                output of segmentation network
            ground_truth : tensor
                groundtruth segmentation
            scalar_dictionary : dict
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

            # Logging segmentation outputs to summary

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
            gt_height, gt_width = ground_truth.shape[-2:]
            if gt_height != n_height or gt_width != n_width:
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

            summary_writer.add_image('input_output_groundtruth_error',
                torchvision.utils.make_grid(display_summary, nrow=1),
                global_step=step)

            # Log scalars
            for (metric_name, metric_value) in scalar_dictionary.items():
                summary_writer.add_scalar(metric_name, metric_value, global_step=step)
