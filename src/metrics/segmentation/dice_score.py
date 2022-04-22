import torch


BASE_NUM_KERNELS = 64
EPS = 1e-9

def dice(prediction, truth):
    return 2.0 * torch.sum(prediction * truth, dim=[1, 2, 3, 4]) / (torch.sum((prediction ** 2 + truth ** 2), [1, 2, 3, 4]) + EPS)

def dice_2(prediction, truth):
    return 2.0 * torch.sum(prediction * truth, dim=[0, 1, 2]) / (torch.sum((prediction ** 2 + truth ** 2), [0, 1, 2]) + EPS)



def dice_score(prediction, truth):
    if len(prediction.shape) > 4:
        dc = dice(prediction, truth)
    else:
        dc = dice_2(prediction, truth)
    return torch.mean(dc, dim=0)