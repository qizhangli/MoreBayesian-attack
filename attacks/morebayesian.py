import numpy as np
import torch
import torch.nn.functional as F
import random

from .helper import random_start_function, to_np_uint8, update_and_clip
from torchvision import transforms as T
from PIL import Image

def get_input_grad(x, y, model_list):
    ce_grad_sum = 0
    loss_sum = 0
    for model in model_list:
        x.requires_grad_(True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        ce_grad_sum += torch.autograd.grad(loss, [x, ])[0].data
        loss_sum += loss.item()
    final_grad = ce_grad_sum / len(model_list)
    loss_avg = loss_sum / len(model_list)
    return final_grad, loss_avg


def morebayesian_attack(args, ori_img, label, mean_model, sqmean_model, get_model_list, verbose=True):
    batch_size = len(ori_img)
    att_img = ori_img.clone()
    for i in range(args.niters):
        model_list = get_model_list(args, mean_model, sqmean_model)
        input_grad, loss = get_input_grad(att_img, label, model_list)
        att_img = update_and_clip(ori_img, att_img, input_grad, args.epsilon, args.step_size, args.constraint)
        if verbose:
            print('iter {}, loss {:.4f}'.format(i, loss), end='\n')
    return att_img
