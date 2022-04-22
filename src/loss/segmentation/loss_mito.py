
import torch
import torch.nn as nn

BASE_NUM_KERNELS = 64
EPS = 1e-9

def dice(prediction, truth):
    return 2.0 * torch.sum(prediction * truth, dim=[1, 2, 3, 4]) / (torch.sum((prediction ** 2 + truth ** 2), [1, 2, 3, 4]) + EPS)

def dice_2(prediction, truth):
    return 2.0 * torch.sum(prediction * truth, dim=[0, 1, 2, 3]) / (torch.sum((prediction ** 2 + truth ** 2), [0, 1, 2, 3]) + EPS)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device = None,
            dtype = None,
            eps = 1e-6) -> torch.Tensor:
    """Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 4:
        raise ValueError("Invalid depth shape, we expect BxHxWxD. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width, depth = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width, depth,
                          device=device, dtype=dtype)
    # one_hot = torch.zeros_like(labels, device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def dice_loss_2(
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    eps: float = 1e-6
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxNxHxWxD. Got: {}"
                            .format(input.shape))
    if not input.shape[-3:] == target.shape[-3:]:
        raise ValueError("input and target shapes must be the same. Got: {}"
                            .format(input.shape, input.shape))
    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input.device, target.device))
    # compute softmax over the classes axis
    input_soft = Fn.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot = one_hot(target.to(torch.int64).squeeze(1), num_classes=input.shape[1],
                                device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3, 4)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(1. - dice_score)



def dice_score(prediction, truth):

    if len(prediction.shape) > 4:
        dc = dice(prediction, truth)
    else:
        dc = dice_2(prediction, truth)
    return torch.mean(dc, dim=0)

def dice_score_2(prediction, truth):
    prediction =torch.sigmoid(prediction)
    dc = dice(prediction, truth)
    return dc

def dice_loss(prediction, truth):
    dc = dice(prediction, truth)
    return torch.mean(1.0 - dc, dim=0)



class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        # y_true = torch.squeeze(y_true, dim=1).long()
        return self.loss(y_pred, y_true)


class DiceCELoss_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=False, softmax=True)
        # self.cross_entropy = CrossEntropyLoss()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=None,
            reduction="mean",
        )

    # def forward(self, y_pred, y_true):
    #     dice = self.dice(y_pred, y_true)
    #     cross_entropy = self.cross_entropy(y_pred, y_true)
    #     return dice + cross_entropy


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is nither 1 or the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)

        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        ce_loss = self.cross_entropy(input, target)
        total_loss: torch.Tensor = dice_loss + ce_loss
        return total_loss

class DiceCELoss_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
        # self.dice = DiceLoss(to_onehot_y=False, softmax=False)
        self.dice = dice_loss
        self.bce = nn.BCELoss()
        

    def forward(self, input, target):
        input = self.act(input)
        dice_loss = self.dice(input, target)
        bce_loss = self.bce(input, target)
        loss = 0.5*dice_loss + 0.5*bce_loss
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
        # self.dice = DiceLoss(to_onehot_y=False, softmax=True)
        self.bce = nn.BCELoss()
        

    def forward(self, input, target):
        # dice_loss = self.dice(input, target)
        act = self.act(input)
        bce_loss = self.bce(act, target)
        # loss = 0.5*dice_loss + 0.5*bce_loss
        return bce_loss