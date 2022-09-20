import os
import pdb
import torch
from torch import nn
import math
import numpy as np
from configs.phyre_cfg import _C as cfg



def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim

    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    img_height = img_tensor.shape[2]
    img_width = img_tensor.shape[3]
    num_channels = img_tensor.shape[4]
    a = torch.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = a.permute(0,1,2,4,3,5,6)
    patch_tensor = torch.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor



def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = patch_tensor.shape[0]
    seq_length = patch_tensor.shape[1]
    patch_height = patch_tensor.shape[2]
    patch_width = patch_tensor.shape[3]
    channels = patch_tensor.shape[4]
    img_channels = channels // (patch_size*patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = a.permute(0,1,2,4,3,5,6)
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor

def reserve_schedule_sampling_exp(itr,bs,args):  #100,100,20:25000
    r_sampling_step_1 = 5000
    r_sampling_step_2 = 15000
    r_exp_alpha = 1000
    total_length = args.total_length
    input_length = args.input_length
    if itr < r_sampling_step_1:
        r_eta = 0.5
    elif itr < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - r_sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0
    #FIXME:
    if itr < r_sampling_step_1:
        eta = 0.5
    elif itr < r_sampling_step_2:
        eta = 0.5 - (0.5 / (r_sampling_step_2 - r_sampling_step_1)) * (itr - r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (bs, input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (bs, total_length - input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((cfg.INPUT.INPUT_WIDTH // args.patch_size,
                    cfg.INPUT.INPUT_WIDTH // args.patch_size,
                    args.patch_size ** 2 * 3))
    zeros = np.zeros((cfg.INPUT.INPUT_WIDTH // args.patch_size,
                      cfg.INPUT.INPUT_WIDTH // args.patch_size,
                      args.patch_size ** 2 * 3))

    real_input_flag = []
    for i in range(bs):
        for j in range(total_length - 2):
            if j < input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = torch.tensor(np.array(real_input_flag))
    real_input_flag = torch.reshape(real_input_flag,
                                 (bs,
                                  total_length - 2,
                                  cfg.INPUT.INPUT_WIDTH // args.patch_size,
                                  cfg.INPUT.INPUT_WIDTH // args.patch_size,
                                  args.patch_size ** 2 * 3))
    return real_input_flag

