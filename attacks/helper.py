import numpy as np
import torch


def update_and_clip(ori_img, att_img, grad, epsilon, step_size, norm):
    if norm == "linf":
        att_img = att_img.data + step_size * torch.sign(grad)
        att_img = torch.where(att_img > ori_img + epsilon, ori_img + epsilon, att_img)
        att_img = torch.where(att_img < ori_img - epsilon, ori_img - epsilon, att_img)
        att_img = torch.clamp(att_img, min=0, max=1)
    elif norm == "l2":
        grad = grad / (grad.norm(p=2,dim = (1,2,3), keepdim=True) + 1e-12)
        att_img = att_img.data + step_size * grad
        l2_perturb = att_img - ori_img
        l2_perturb = l2_perturb.renorm(p=2, dim = 0, maxnorm=epsilon)
        att_img = ori_img + l2_perturb
        att_img = torch.clamp(att_img, min=0, max=1)
    return att_img


def random_start_function(x, epsilon):
    x_p = x + x.new(x.size()).uniform_(-epsilon, epsilon)
    x_p = torch.clamp(x_p, min=0, max=1)
    return x_p


def to_np_uint8(x):
    return torch.round(x.data*255).cpu().numpy().astype(np.uint8())
