import argparse
import logging
import os
import sys

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tvF
from torch.backends import cudnn

import models as MODEL

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--model_dir', type=str, default=None)
args = parser.parse_args()

if not os.path.isdir("test_logs"):
    os.mkdir("test_logs")

logging.basicConfig(filename='test_logs/{}.log'.format(args.dir.split('/')[-1]), level=logging.INFO)

logging.info(args)


cudnn.benchmark = False
cudnn.deterministic = True
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def normalize(x, ms=None):
    if ms == None:
        ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - ms[0][i]) / ms[1][i]
    return x

def load_img(img, trans):
    img_pil = tvF.to_pil_image(img)
    return trans(img_pil)

class NumpyImages(torch.utils.data.Dataset):
    def __init__(self, npy_dir, transforms=None):
        super(NumpyImages, self).__init__()
        npy_ls = []
        for npy_name in os.listdir(npy_dir):
            if npy_name[:5] == 'batch':
                npy_ls.append(npy_name)
        self.data = []
        for npy_ind in range(len(npy_ls)):
            self.data.append(np.load(npy_dir + '/batch_{}.npy'.format(npy_ind)))
        self.data = np.concatenate(self.data, axis=0)
        self.target = np.load(npy_dir+"/labels.npy")
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float() / 255, self.target[index]
    
    def __len__(self,):
        return len(self.target)


dataset = NumpyImages(args.dir, transforms=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=False, num_workers=4)

def test(model, trans, dataloader=dataloader):
    img_num = 0
    count = 0
    dir_ls = os.listdir(args.dir)
    for img, label in dataloader:
        label = label.to(device)
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(trans(img)), dim=1).view(1,-1)
        count += (label != pred.squeeze(0)).sum().item()
        img_num += len(img)
    return round(100. * count / img_num, 2)


trans_pnas = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Resize((331, 331)),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
trans_se = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trans_incep = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Resize((299, 299)),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


VGG19 = torchvision.models.vgg19_bn()
state_dict = torch.load(args.model_dir + 'pretrained_models/imagenet/vgg19_bn-c79401a0.pth', map_location = device)
VGG19.load_state_dict(state_dict)
VGG19.to(device)
VGG19.eval()
vgg_fr = test(model = VGG19, trans = trans_se)
logging.info(('VGG19:', vgg_fr))
del VGG19

resnet152 = torchvision.models.resnet152()
state_dict = torch.load(args.model_dir + 'pretrained_models/imagenet/resnet152-b121ed2d.pth')
resnet152.load_state_dict(state_dict)
resnet152.to(device)
resnet152.eval()
resnet152_fr = test(model = resnet152, trans = trans_se)
logging.info(('resnet152:', resnet152_fr))
del resnet152

inceptionv3 = MODEL.inceptionv3.Inception3()
inceptionv3.to(device)
inceptionv3.load_state_dict(torch.load(args.model_dir + 'pretrained_models/imagenet/inception_v3_google-1a9a5a14.pth'))
inceptionv3.eval()
logging.info(('inceptionv3:', test(model = inceptionv3, trans = trans_incep)))
del inceptionv3

densenet = torchvision.models.densenet121(pretrained=False)
densenet.to(device)
import re

pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

state_dict = torch.load(args.model_dir + 'pretrained_models/imagenet/densenet121-a639ec97.pth')
for key in list(state_dict.keys()):
    res = pattern.match(key)
    if res:
        new_key = res.group(1) + res.group(2)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
densenet.load_state_dict(state_dict)
densenet.eval()
logging.info(('densenet:', test(model = densenet, trans = trans_se)))
del densenet

mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
mobilenet.to(device)
mobilenet.load_state_dict(torch.load(args.model_dir + 'pretrained_models/imagenet/mobilenet_v2-b0353104.pth'))
mobilenet.eval()
logging.info(('mobilenet:', test(model = mobilenet, trans = trans_se)))
del mobilenet

senet = MODEL.senet.senet154(ckpt_dir =args.model_dir + 'pretrained_models/imagenet/senet154-c7b49a05.pth')
senet.to(device)
senet.eval()
logging.info(('senet:', test(model = senet, trans = trans_se)))
del senet

resnext = torchvision.models.resnext101_32x8d()
state_dict = torch.load(args.model_dir + 'pretrained_models/imagenet/resnext101_32x8d-8ba56ff5.pth', map_location = device)
resnext.load_state_dict(state_dict)
resnext.to(device)
resnext.eval()
logging.info(('resnext:', test(model = resnext, trans = trans_se)))
del resnext

WRN = torchvision.models.wide_resnet101_2()
state_dict = torch.load(args.model_dir + 'pretrained_models/imagenet/wide_resnet101_2-32ee1156.pth', map_location = device)
WRN.load_state_dict(state_dict)
WRN.to(device)
WRN.eval()
wrn_fr = test(model = WRN, trans = trans_se)
logging.info(('WRN:', wrn_fr))
del WRN

pnasnet = MODEL.pnasnet.pnasnet5large(ckpt_dir =args.model_dir + 'pretrained_models/imagenet/pnasnet5large-bf079911.pth', num_classes=1000, pretrained='imagenet')
pnasnet.to(device)
pnasnet.eval()
logging.info(('pnasnet:', test(model = pnasnet, trans = trans_pnas)))
del pnasnet

mnasnet = torchvision.models.mnasnet1_0()
state_dict = torch.load(args.model_dir + 'pretrained_models/imagenet/mnasnet1.0_top1_73.512-f206786ef8.pth', map_location = device)
mnasnet.load_state_dict(state_dict)
mnasnet.to(device)
mnasnet.eval()
mnas_fr = test(model = mnasnet, trans = trans_se)
logging.info(('mnasnet:', mnas_fr))
del mnasnet
