import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from evaluation import *

#############################################################################################
# Graph-Tensor Encoding Losses
#
#############################################################################################
def SoftmaxOutput(image_graph, MAX_DEGREE=6):
    # Initialize output image graph
    new_outputs = torch.zeros_like(image_graph)
    
    # Normalize nodes positions
    new_outputs[:,0] = F.sigmoid(image_graph[:,0] - image_graph[:,1])
    new_outputs[:,1] = 1. - new_outputs[:,0]
    
    # Normalize neighbors positions and directions
    for i in range(MAX_DEGREE):
        new_outputs[:,2+i*4] = F.sigmoid(image_graph[:,2+i*4] - image_graph[:,2+i*4+1])
        new_outputs[:,2+i*4+1] = 1. - new_outputs[:,2+i*4]
        new_outputs[:,2+i*4+2:2+i*4+4] = image_graph[:,2+i*4+2:2+i*4+4]
    
    return new_outputs


class SupervisedLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SupervisedLoss, self).__init__()
    
    def forward(self, imagegraph_outputs, imagegraph_targets, MAX_DEGREE=6):

        # prepare soft masks
        soft_mask = torch.clamp(imagegraph_targets[:,0]-0.01, 0.0, 0.99)
        soft_mask = soft_mask + 0.01
        soft_mask2 = torch.reshape(soft_mask, [imagegraph_outputs.shape[0], imagegraph_outputs.shape[2], imagegraph_outputs.shape[3]])

        # Keypoints prob loss
        keypoint_prob_loss = F.cross_entropy(imagegraph_outputs[:,0:2], imagegraph_targets[:,0:2], reduction='mean')

        # direction prob loss
        direction_prob_loss = 0
        for i in range(MAX_DEGREE):
            prob_output = imagegraph_outputs[:, 2 + i*4 : 2 + i*4 + 2]
            prob_target = imagegraph_targets[:, 2 + i*4 : 2 + i*4 + 2]
            # only at key points! 
            direction_prob_loss += torch.mean(torch.multiply((soft_mask2), F.cross_entropy(prob_output, prob_target, reduction='none')))
        direction_prob_loss /= MAX_DEGREE

        # direction vector loss 
        direction_vector_loss = 0
        for i in range(MAX_DEGREE):
            vector_output = imagegraph_outputs[:, 2 + i*4 + 2 : 2 + i*4 + 4]
            vector_target = imagegraph_targets[:, 2 + i*4 + 2 : 2 + i*4 + 4]
            # only at key points! 
            direction_vector_loss += torch.mean(torch.multiply(torch.unsqueeze(soft_mask, dim=1), torch.square(vector_output - vector_target)))
        direction_vector_loss /= MAX_DEGREE 

        return keypoint_prob_loss, direction_prob_loss* 10.0, direction_vector_loss * 1000.0


def NormalizeOutput(image_graph, MAX_DEGREE=6):
    # Normalize nodes positions
    image_graph[:,0] = F.sigmoid(image_graph[:,0])
    
    # Normalize neighbors positions and directions
    for i in range(MAX_DEGREE):
        image_graph[:, 3*i + 1] = F.sigmoid(image_graph[:, 3*i + 1])
        image_graph[:, 3*i + 2 : 3*i + 4] = F.tanh(image_graph[:, 3*i + 2 : 3*i + 4])
    
    return image_graph


class SupervisedLossCompact(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SupervisedLossCompact, self).__init__()
    
    def forward(self, imagegraph_outputs, imagegraph_targets, MAX_DEGREE=6):

        # prepare soft masks
        soft_mask = torch.clamp(imagegraph_targets[:,0]-0.01, 0.0, 0.99)
        soft_mask = soft_mask + 0.01
        soft_mask2 = torch.reshape(soft_mask, [imagegraph_outputs.shape[0], imagegraph_outputs.shape[2], imagegraph_outputs.shape[3]])

        # keypoints prob loss
        keypoint_prob_loss = F.binary_cross_entropy(imagegraph_outputs[:,0], imagegraph_targets[:,0], reduction='mean')

        # neighbors prob and vector loss
        direction_prob_loss, direction_vector_loss = 0, 0
        for i in range(MAX_DEGREE):
            # prob arrays
            prob_output = imagegraph_outputs[:, 3*i + 1]
            prob_target = imagegraph_targets[:, 3*i + 1]
            # vector arrays
            vector_output = imagegraph_outputs[:, 3*i + 2 : 3*i + 4]
            vector_target = imagegraph_targets[:, 3*i + 2 : 3*i + 4]
            # only at key points!
            direction_prob_loss += torch.mean(torch.multiply(soft_mask2, F.binary_cross_entropy(prob_output, prob_target, reduction='none')))
            direction_vector_loss += torch.mean(torch.multiply(torch.unsqueeze(soft_mask, dim=1), torch.square(vector_output - vector_target)))
        direction_prob_loss /= MAX_DEGREE
        direction_vector_loss /= MAX_DEGREE

        return keypoint_prob_loss, direction_prob_loss* 10.0, direction_vector_loss * 100.0


#############################################################################################
# Segmentation Losses
#
#############################################################################################
class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.7, gamma=4/3):
        
        # flatten label and prediction tensors
        d_loss = dice_loss(inputs, targets)
        ft_loss = focal_tversky_loss(inputs, targets, alpha=alpha, gamma=gamma)
        
        # total loss
        return d_loss + ft_loss


#############################################################################################
# Learning rate scheduler
#
#############################################################################################
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler():
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose

    def get_batch_lr(self):
        self.global_step = self.global_step + 1
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        return lr
