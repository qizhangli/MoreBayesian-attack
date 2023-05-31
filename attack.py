import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.backends import cudnn
from torch.utils.data import DataLoader

from attacks.helper import to_np_uint8
import copy
import torch.optim as optim
from collections import OrderedDict
from utils import build_dataset, build_model, cat_grad
from dataset import SelectedImagenet
from attacks.morebayesian import morebayesian_attack

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--step-size', type=float, default=1)
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--force', default=False, action="store_true")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--constraint', type=str, default="linf", choices=["linf", "l2"])
parser.add_argument('--n_models', type=int, default=20)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--scale', type=float, default=1.5)
parser.add_argument('--source-model-dir', type=str, default=None)
parser.add_argument('--data-info-dir', type=str, default=None)
parser.add_argument('--data-dir', type=str, default=None)
parser.add_argument('--save-dir', type=str, default=None)
args = parser.parse_args()

def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.
    print(args)
    
    os.makedirs(args.save_dir, exist_ok=True if args.force else False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_loader = build_dataset(args)

    state_dict = torch.load(args.source_model_dir)
    mean_model = build_model(args, device, state_dict["mean_state_dict"])
    sqmean_model = build_model(args, device, state_dict["sqmean_state_dict"])
    mean_model = nn.DataParallel(mean_model)
    
    def get_model_list(args, mean_model, sqmean_model):
        model_list = []
        for model_ind in range(args.n_models):
            model_list.append(copy.deepcopy(mean_model))
            noise_dict = OrderedDict()
            for (name, param_mean), param_sqmean, param_cur in zip(mean_model.named_parameters(), sqmean_model.parameters(), model_list[-1].parameters()):
                var = torch.clamp(param_sqmean.data - param_mean.data**2, 1e-30)
                var = var + args.beta
                noise_dict[name] = var.sqrt() * torch.randn_like(param_mean, requires_grad=False)
            for (name, param_cur), (_, noise) in zip(model_list[-1].named_parameters(), noise_dict.items()):
                param_cur.data.add_(noise, alpha=args.scale)
        return model_list
    
        
    print("Models Load Complete. Start Attack...")
    
    # ATTACK
    label_ls = []
    for ind, (ori_img, label) in enumerate(data_loader):
        label_ls.append(label)
        ori_img, label = ori_img.to(device), label.to(device)
        img_adv = morebayesian_attack(args, ori_img, label, mean_model, sqmean_model, get_model_list)
        np.save(os.path.join(args.save_dir, 'batch_{}.npy'.format(ind)), to_np_uint8(img_adv))
        print(' batch_{}.npy saved'.format(ind))
    label_ls = torch.cat(label_ls)
    np.save(os.path.join(args.save_dir, 'labels.npy'), label_ls.numpy())
    print('images saved')

if __name__ == '__main__':
    main()
