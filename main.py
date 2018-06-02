from model import *
from trainer import Trainer
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./standard_infogan/', help='folder to output images and model checkpoints')
parser.add_argument('--ndiscrete', type=int, default=10, help='size of the latent z vector')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


fe = FrontEnd()
d = D()
q = Q(opt)
g = G()
for i in [fe, d, q, g]:
  i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q,opt)
trainer.train()
