import argparse
import copy
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.backends import cudnn
from torch.utils.data import DataLoader

from attacks.helper import to_np_uint8
from utils import (RandomResizedCrop, _concat, assign_grad, get_grad, 
                   update_bn_imgnet, update_swag_model, eval_imgnet, 
                   add_into_weights, cat_grad)


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--swa-start', type=int, default=0)
parser.add_argument('--swa-n', type=int, default=300)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data-path', default=None, type=str)
parser.add_argument('--save-dir', type=str, default=None)
args = parser.parse_args()


def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    args.constraint = "linf"
    args.niters = 10
    args.epsilon = 8 / 255.
    args.step_size = 1 / 255.
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info("Input args: %r", args)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    transform_train = T.Compose([
            RandomResizedCrop(224, interpolation=3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_train = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    
    model = torchvision.models.resnet50(pretrained=True)
    model = nn.DataParallel(model)
    model = model.cuda()
    proxy = copy.deepcopy(model)
    
    print("SWAG training ...")
    mean_model = copy.deepcopy(model)
    sqmean_model = copy.deepcopy(model)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    n_collected = 0
    n_ensembled = 0
    for epoch in range(args.epochs):
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            model.train()
            
            output_cln = model(img)
            loss_normal = F.cross_entropy(output_cln, label)
            optimizer.zero_grad()
            loss_normal.backward()
            grad_normal = get_grad(model)
            norm_grad_normal = cat_grad(grad_normal).norm()

            add_into_weights(model, grad_normal, gamma = +0.1 / (norm_grad_normal+1e-20))
            loss_add = F.cross_entropy(model(img), label)
            optimizer.zero_grad()
            loss_add.backward()
            grad_add = get_grad(model)
            add_into_weights(model, grad_normal, gamma = -0.1 / (norm_grad_normal+1e-20))
            
            optimizer.zero_grad()
            grad_new_dict = OrderedDict()
            for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                grad_new_dict[name] = g_normal + (args.lam / 0.1) * (g_add - g_normal) 
            assign_grad(model, grad_new_dict)
            optimizer.step()
            
            if i % 100 == 0:
                acc = 100*(output_cln.argmax(1) == label).sum() / len(img)
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.2f}'.format(
                    epoch, i * len(img), len(train_loader.dataset),
                        100. * i / len(train_loader), loss_normal.item(), acc))
            
            # SWAG
            if ((epoch+1) > args.swa_start 
                and ((epoch - args.swa_start)*len(train_loader)+i) % (((args.epochs-args.swa_start)*len(train_loader)) // args.swa_n) == 0):
                update_swag_model(model, mean_model, sqmean_model, n_ensembled)
                n_ensembled+=1
                
        loss_cln_eval, acc_eval = eval_imgnet(args, val_loader, model, device)
        logging.info('CURRENT EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))
        print("updating BN statistics ... ")
        update_bn_imgnet(train_loader, mean_model)
        loss_cln_eval, acc_eval = eval_imgnet(args, val_loader, mean_model, device)
        logging.info('SWA EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))
        
        torch.save({"state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                    os.path.join(args.save_dir, 'ep_{}.pt'.format(epoch)))
        torch.save({"mean_state_dict": mean_model.state_dict(),
                    "sqmean_state_dict": sqmean_model.state_dict(),
                    "epoch": epoch},
                    os.path.join(args.save_dir, 'swag_ep_{}.pt'.format(epoch)))
    

if __name__ == '__main__':
    main()
