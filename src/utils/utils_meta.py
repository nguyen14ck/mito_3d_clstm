import os
import random
import functools
from collections import OrderedDict


import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
import numpy as np
import torch


def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad


def set_seed_x(seed):
    # for reproducibility.
    # note that pytorch is not completely reproducible
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def set_seed(seed_value, use_cuda=True):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    scipy.stats.multivariate_normal.random_state = seed_value
    torch.initial_seed() # dataloader multi processing
    torch.manual_seed(seed_value) # cpu  vars
    # torch.set_deterministic(True)
    
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # multi gpus
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = True




def set_gpu(x):
    x = [str(e) for e in x]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(x)
    print('using gpu:', ','.join(x))

def check_dir(args):
    # save path
    path = os.path.join(args.result_path, args.alg)
    if not os.path.exists(path):
        os.makedirs(path)
    return None

# https://github.com/sehkmg/tsvprint/blob/master/utils.py
def dict2tsv(res, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')

class BestTracker:
    '''Decorator for train function.
       Get ordered dict result (res),
       track best dice coef (self.best_dice & best epoch (self.best_epoch) and
       append them to ordered dict result (res).
       Also, save the best result to file (best.txt).
       Return ordered dict result (res).'''
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.best_epoch = 0
        self.best_test_dice= 0

    def __call__(self, *args, **kwargs):
        res = self.func(*args, **kwargs)

        if res['test_dice'] > self.best_test_dice:
            self.best_epoch = res['epoch']

            self.best_test_dice = res['test_dice']
            is_best = True
        else:
            is_best = False

        res['best_epoch'] = self.best_epoch

        res['best_test_dice'] = self.best_test_dice

        return res, is_best



        def get_confusion_matrix_elements(groundtruth_list, predicted_list):
            """returns confusion matrix elements i.e TN, FP, FN, TP as floats

            """
            predicted_list = np.round(predicted_list).astype(int)
            groundtruth_list = np.round(groundtruth_list).astype(int)
            groundtruth_list=groundtruth_list.reshape(-1)
            predicted_list=predicted_list.reshape(-1)
            tn, fp, fn, tp = confusion_matrix(groundtruth_list, predicted_list).ravel()
            tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

            return tn, fp, fn, tp

        def get_mcc(groundtruth_list, predicted_list):
            """Return mcc covering edge cases"""

            tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
            mcc = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            return mcc

        def get_precision(groundtruth_list, predicted_list):

            tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total

            return accuracy

        ################
        def log_images (x,  y_pred,  y_true=None, channel=1):
            images = []
            x_np = x[:, channel].cpu().numpy()
            y_true_np = y_true[:, 0].cpu().numpy()
            y_pred_np = y_pred[:, 0].cpu().numpy()
            for i in range(x_np.shape[0]):
                image = gray2rgb(np.squeeze(x_np[i]))
                image = outline(image, y_pred_np[i], color=[255, 0, 0])
                image = outline(image, y_true_np[i], color=[0, 255, 0])
                images.append(image)
            return images



        def gray2rgb(image):
            w, h = image.shape
            image += np.abs(np.min(image))
            image_max = np.abs(np.max(image))
            if image_max > 0:
                image /= image_max
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
            return ret


        def outline(image, mask, color):
            mask = np.round(mask)
            yy, xx = np.nonzero(mask)
            for y, x in zip(yy, xx):
                if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
                    image[max(0, y) : y + 1, max(0, x) : x + 1] = color
            return image
