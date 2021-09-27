import torch


def cross_entropy_loss_func(src, tgt, w=None):
    '''
    Computes cross entropy loss between source and target with weighting

    Arg(s):
        src : tensor[float32]
            source logits tensor (N x K x H x W)
        tgt : tensor[long]
            target ground truth (N x H x W)
        w : tensor[float32]
            K element vector or weight map corresponding to weight of each class
    Returns:
        float32 : cross entropy loss
    '''

    if w is None:
        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        return loss(src, tgt)

    w_n_dims = len(w.size())

    if w_n_dims == 1:
        # Using weight vector
        loss = torch.nn.CrossEntropyLoss(weight=w, reduction='mean')
        return loss(src, tgt)
    else:
        # Using a weight map
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        return torch.mean(loss(src, tgt) * w)

def binary_cross_entropy_loss_func(src, tgt, w=None):
    '''
    Computes binary cross entropy loss between source and target with weighting

    Arg(s):
        src : tensor[float32]
            source logits tensor (N x H x W)
        tgt : tensor[float32]
            target ground truth (N x H x W)
        w : tensor[float32]
            1 element positive weight or binary weight map corresponding to weight of each class
    Returns:
        float32 : binary cross entropy loss
    '''

    # Just in case int64 or long type is provided
    tgt = tgt.to(torch.float32)

    if w is None:
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        return loss(src, tgt)

    w_n_dims = len(w.size())

    if w_n_dims == 1:
        # Using a weight for positive class
        loss = torch.nn.BCEWithLogitsLoss(
            reduction='mean',
            pos_weight=w[-1])
        return loss(src, tgt)
    else:
        # Using a weight map
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        return torch.mean(loss(src, tgt) * w)

def soft_dice_loss_func(src, tgt, smoothing=1.0):
    '''
    Computes soft dice loss between source and target

    Arg(s):
        src : tensor[float32]
            source logits tensor (N x H x W)
        tgt : tensor[float32]
            target ground truth (N x H x W)
        smoothing : float32
            1 element offset for numerical stability
    Returns:
        float32 : cross entropy loss
    '''

    intersection = torch.sum(src * tgt)
    numerator = 2 * intersection + smoothing
    denominator = torch.sum(src) + torch.sum(tgt) + smoothing

    return 1 - (numerator / denominator)
