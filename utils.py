import copy
import csv
import math
import os
from collections import OrderedDict

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)
    def __len__(self):
        return len(self.selected_list)


def build_dataset(args):
    img_transform = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor()
        ])
    dataset = SelectedImagenet(imagenet_val_dir=args.data_dir,
                               selected_images_csv=args.data_info_dir,
                               transform=img_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory = True)
    return data_loader


def build_model(args, device, state_dict=False):
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))
    model = torchvision.models.resnet50()
    if not state_dict:
        model.load_state_dict(torch.load(os.path.join(args.source_model_dir, 'resnet50-19c8e357.pth'), map_location='cpu'))
    else:
        if "module" in list(state_dict.keys())[0]:
            model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.module
        else:
            model.load_state_dict(state_dict)
        model.eval()
    model = nn.Sequential(normalize, model)
    model.to(device)
    return model


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class RandomResizedCrop(T.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = torchvision.transforms.functional.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w


def update_swag_model(model, mean_model, sqmean_model, n):
    for param, param_mean, param_sqmean in zip(model.parameters(), mean_model.parameters(), sqmean_model.parameters()):
        param_mean.data.mul_(n / (n+1.)).add_(param, alpha=1./(n+1.))
        param_sqmean.data.mul_(n / (n+1.)).add_(param**2, alpha=1./(n+1.))


def update_bn_imgnet(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    if not momenta:
        return
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0
    for i, input in enumerate(loader):
        # using 10% of the training data to update batch-normalization statistics
        if i > len(loader)//10:
            break
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)
        with torch.no_grad():
            model(input)
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
    



def add_into_weights(model, grad_on_weights, gamma):
    names_in_gow = grad_on_weights.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_gow:
                param.add_(gamma * grad_on_weights[name])


def get_grad(model):
    grad_dict = OrderedDict()
    for name, param in model.named_parameters():
        grad_dict[name] = param.grad.data+0
    return grad_dict


def assign_grad(model, grad_dict):
    names_in_grad_dict = grad_dict.keys()
    for name, param in model.named_parameters():
        if name in names_in_grad_dict:
            if param.grad != None:
                param.grad.data.mul_(0).add_(grad_dict[name])
            else:
                param.grad = grad_dict[name]


def cat_grad(grad_dict):
    dls = []
    for name, d in grad_dict.items():
        dls.append(d)
    return _concat(dls)


def eval_imgnet(args, val_loader, model, device):
    loss_eval = 0
    # grad_norm_eval = 0
    acc_eval = 0
    for i, (img, label) in enumerate(val_loader):
        img, label = img.to(device), label.to(device)
        model.eval()
        with torch.no_grad():
            output = model(img)
        loss = F.cross_entropy(output, label)
        acc = 100*(output.argmax(1) == label).sum() / len(img)
        loss_eval+=loss.item()
        acc_eval+=acc
        if i == 4:
            loss_eval/=(i+1)
            acc_eval/=(i+1)
            break
    return loss_eval, acc_eval
