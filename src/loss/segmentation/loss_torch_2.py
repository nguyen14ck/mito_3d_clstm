# import tensorflow as tf
# import keras.backend as K
# from keras.losses import binary_crossentropy

import torch
import torch.nn as nn
import torch.nn.functional as F


beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
K_EPSILON = 1e-7
K_BACKEND_EPSILON = 1e-7
# torch.finfo(torch.float32).eps
smooth = 1


class Semantic_loss_functions(object):
    def __init__(self, activation=None, bce=False):
        self.activation = None
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        
        if bce:
            self.bce_loss = nn.BCELoss()
        else:
            self.bce_loss = None


    def dice_coef(self, y_pred, y_true):
        if self.activation is not None:
            y_pred = self.activation(y_pred)
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + K_EPSILON) / (
                    torch.sum(y_true_f) + torch.sum(y_pred_f) + K_EPSILON)

    def dice_coefficient(self, prediction, truth):
        if self.activation is not None:
            prediction = self.activation(prediction)
            # prediction = torch.argmax(prediction, dim=1)
        dims = list(range(1, len(truth.shape)))
        upper = 2.0 * torch.sum(prediction * truth, dim=dims) + 1
        # lower = torch.sum((prediction ** 2 + truth ** 2), dim=dims) + K_EPSILON
        lower = torch.sum((prediction + truth), dim=dims) + 1
        return upper / lower

    def sensitivity(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K_EPSILON)

    def specificity(self, y_pred, y_true):
        true_negatives = torch.sum(
            torch.round(torch.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = torch.sum(torch.round(torch.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K_EPSILON)

    def convert_to_logits(self, y_pred):
        y_pred = torch.clip_by_value(y_pred, K_BACKEND_EPSILON,
                                  1 - K_BACKEND_EPSILON)
        return torch.log(y_pred / (1 - y_pred))

    def weighted_cross_entropyloss(self, y_pred, y_true):
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = torch.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return torch.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (torch.log1p(torch.exp(-torch.abs(logits))) + torch.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, K_BACKEND_EPSILON,
                                  1 - K_BACKEND_EPSILON)
        logits = torch.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)

        return torch.mean(loss)

    def depth_softmax(self, matrix):
        sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / torch.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coef(self, y_pred, y_true):
        if self.activation is not None:
            y_pred = self.activation(y_pred)
        smooth = 1.
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score

    def generalized_dice_coefficient(self, y_pred, y_true, batch=True, reduction=None):
        if self.activation is not None:
            y_pred = self.activation(y_pred)
        smooth = 1.
        # y_true_f = torch.flatten(y_true)
        # y_pred_f = torch.flatten(y_pred)
        # intersection = torch.sum(y_true_f * y_pred_f)
        # score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        # return score
        if batch:
            dims = list(range(1, len(y_true.shape)))
        else:
            dims = list(range(len(y_true.shape)))
        upper = 2.0 * torch.sum(y_pred * y_true, dim=dims) + smooth
        # lower = torch.sum((y_pred ** 2 + y_true ** 2), dim=dims) + K_EPSILON
        lower = torch.sum((y_pred + y_true), dim=dims) + smooth
        dice = upper/lower
        if reduction == 'mean':
            return dice.mean()
        return dice

    def dice_loss(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        loss = 1 - self.dice_coefficient(y_pred, y_true)
        return loss


    def dice_loss_2(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        dice_score = self.dice_coefficient(y_pred, y_true)
        loss = 1 - dice_score
        return loss, dice_score


    def generalized_dice_loss(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        loss = 1 - self.generalized_dice_coefficient(y_pred, y_true)
        return loss


    def generalized_dice_loss_2(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        dice_score = self.generalized_dice_coefficient(y_pred, y_true)
        loss = 1 - dice_score
        return loss, dice_score

    

    def bce_dice_loss(self, y_pred, y_true):
        if self.activation is not None:
            y_pred = self.activation(y_pred)
        loss = self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)
        return loss / 2.0

    def confusion(self, y_pred, y_true):
        smooth = 1
        y_pred_pos = torch.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = torch.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = torch.sum(y_pos * y_pred_pos)
        fp = torch.sum(y_neg * y_pred_pos)
        fn = torch.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_pred, y_true):
        smooth = 1
        y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
        y_pos = torch.round(torch.clip(y_true, 0, 1))
        tp = (torch.sum(y_pos * y_pred_pos) + smooth) / (torch.sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_pred, y_true):
        smooth = 1
        y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = torch.round(torch.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (torch.sum(y_neg * y_pred_neg) + smooth) / (torch.sum(y_neg) + smooth)
        return tn

    def tversky_index(self, y_pred, y_true):
        if self.activation is not None:
            y_pred = self.activation(y_pred)
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        tvi = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
        return tvi

    def tversky_loss(self, y_pred, y_true, reduction=None):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        tv_loss = torch.tensor(1 - self.tversky_index(y_pred, y_true))
        if reduction == 'mean':
            return tv_loss.mean()
        return tv_loss

    def focal_tversky(self, y_pred, y_true):
        pt_1 = self.tversky_index(y_pred, y_true)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)

    def focal_tversky_loss(self, y_pred, y_true, gamma=0.75):
        pt_1 = self.tversky_index(y_pred, y_true)
        ft_loss = torch.pow((1 - pt_1), gamma)
        # ft_loss = torch.pow((1 - pt_1), gamma)*torch.log(pt_1)
        return ft_loss

    def log_cosh_dice_loss(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        x = self.dice_loss(y_pred, y_true)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


    def log_cosh_dice_loss_2(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        x, dice_score = self.dice_loss_2(y_pred, y_true)
        lcd_loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
        return lcd_loss, dice_score

    
    def log_cosh_generalized_dice_loss(self, y_pred, y_true):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        x = self.generalized_dice_loss(y_pred, y_true)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


    def log_cosh_generalized_dice_loss_2(self, y_pred, y_true, reduction=None):
        # if self.activation is not None:
        #     y_pred = self.activation(y_pred)
        x, dice_score = self.generalized_dice_loss_2(y_pred, y_true)
        lcd_loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
        
        if reduction == 'mean':
            return lcd_loss.mean(), dice_score.mean()

        return lcd_loss, dice_score


    # loss_focal = (1.0 - loss_ce.mul(-1).exp()).pow(self.gamma) * loss_ce