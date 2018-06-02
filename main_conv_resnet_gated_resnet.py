from __future__ import print_function
from trainer import Trainer
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import visdom
from common_net import *
import math
import sys
from model import *
vis = visdom.Visdom()
vis.env = 'conv_gated_resnet_resnet_gen'
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--out_dim', type=int, default=784, help='the output dimension')
parser.add_argument('--nz', type=int, default=74, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=1, help='size of the number of channels in the output')
parser.add_argument('--nsalient', type=int, default=12, help='size of the latent z vector')
parser.add_argument('--ncontinuous', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--ndiscrete', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--nnoise', type=int, default=62, help='size of the latent z vector')
parser.add_argument('--imageSize', type=int, default=28, help='size of the image')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ngf_gate', type=int, default=32)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--ngres', type=int, default=16)
parser.add_argument('--ngres_gate', type=int, default=16)
parser.add_argument('--ndres', type=int, default=4)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--num_samples', type=int, default=1000000, help='number of samples in the dataset')
parser.add_argument('--lambda_info', type=float, default=1, help='Information Regularization Coefficient default=1.0')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate, default=0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./conv_gated_resnet_resnet_gen/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int,default=7, help='manual seed')
parser.add_argument('--dataset', type=str, default='1d', help='which dataset')
opt = parser.parse_args()
opt.nsalient=opt.ncontinuous + opt.ndiscrete
opt.nz = opt.nnoise+opt.ndiscrete+opt.ncontinuous
opt.batchSize= opt.ndiscrete * opt.ndiscrete
print(opt)
torch.manual_seed(7)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True


fe = FrontEnd(opt)
d = D()
q = Q(opt)
g = GatedResnetConvResnetG(opt)

for i in [fe, d, q, g]:
  i.cuda()
  #i.apply(weights_init)

with open(opt.outf+'/architecture.txt' , 'w'  ) as f:
    print(g,file=f)
    print(fe,file=f)
    print(d,file=f)
    print(q,file=f)



trainer = Trainer(g, fe, d, q,opt)
trainer.train()
