import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)

    return logli.sum(1).mean().mul(-1)

class Trainer:

  def __init__(self, G, FE, D, Q,opt):

    self.G = G
    self.FE = FE
    self.D = D
    self.Q = Q
    self.opt = opt
    self.batch_size = opt.batchSize#100

  def _noise_sample(self, dis_c, con_c, noise, bs):

    idx = np.random.randint(self.opt.ndiscrete, size=bs)
    c = np.zeros((bs, self.opt.ndiscrete))
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.normal_(0, 1)
    #con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.opt.nnoise+self.opt.ndiscrete+self.opt.ncontinuous, 1, 1)

    return z, idx

  def train(self):

    real_x = torch.FloatTensor(self.batch_size, self.opt.nc, 28, 28).cuda()
    label = torch.FloatTensor(self.batch_size).cuda()
    dis_c = torch.FloatTensor(self.batch_size, self.opt.ndiscrete).cuda()
    con_c = torch.FloatTensor(self.batch_size, self.opt.ncontinuous).cuda()
    noise = torch.FloatTensor(self.batch_size, self.opt.nnoise).cuda()

    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    dis_c = Variable(dis_c)
    con_c = Variable(con_c)
    noise = Variable(noise)

    criterionD = nn.BCELoss().cuda()
    criterionQ_dis = nn.CrossEntropyLoss().cuda()
    criterionQ_con = log_gaussian()

    optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.0002, betas=(0.5, 0.99))

    dataset = dset.CIFAR10('./cifar_dataset', transform=transforms.Compose([ transforms.CenterCrop(self.opt.imageSize) ,  transforms.ToTensor()]), download=True)
    #dataset= dset.FashionMNIST('./fashion_dataset', transform=transforms.ToTensor(), download=True)
    #dataset= dset.MNIST('./dataset', transform=transforms.ToTensor(),download=True)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    # fixed random variables
    c = np.linspace(-1, 1, self.opt.ndiscrete).reshape(1, -1)
    c = np.repeat(c, self.opt.ndiscrete, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    idx = np.arange(self.opt.ndiscrete).repeat(self.opt.ndiscrete)
    one_hot = np.zeros((self.opt.ndiscrete*self.opt.ndiscrete, self.opt.ndiscrete))
    one_hot[range(self.opt.ndiscrete*self.opt.ndiscrete), idx] = 1
    fix_noise = torch.Tensor(self.opt.ndiscrete*self.opt.ndiscrete, self.opt.nnoise).uniform_(-1, 1)


    for epoch in range(100):

      for num_iters, batch_data in enumerate(dataloader, 0):

        # real part
        optimD.zero_grad()

        x, _ = batch_data

        bs = x.size(0)
        real_x.data.resize_(x.size())
        label.data.resize_(bs)
        dis_c.data.resize_(bs, self.opt.ndiscrete)
        con_c.data.resize_(bs, self.opt.ncontinuous)
        noise.data.resize_(bs, self.opt.nnoise)

        real_x.data.copy_(x)
        fe_out1 = self.FE(real_x)
        probs_real = self.D(fe_out1)
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # fake part
        z, idx = self._noise_sample(dis_c, con_c, noise, bs)
        fake_x = self.G(z)
        fe_out2 = self.FE(fake_x.detach())
        probs_fake = self.D(fe_out2)
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake

        optimD.step()

        # G and Q part
        optimG.zero_grad()

        fe_out = self.FE(fake_x)
        probs_fake = self.D(fe_out)
        label.data.fill_(1.0)

        reconstruct_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = self.Q(fe_out)
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        dis_loss = self.opt.lambda_info * criterionQ_dis(q_logits, target)
        con_loss = self.opt.lambda_info *criterionQ_con(con_c, q_mu, q_var)  # *0.1

        G_loss = reconstruct_loss + dis_loss + con_loss
        G_loss.backward()
        optimG.step()
        Q_loss = dis_loss + con_loss
        if num_iters % 100 == 0:

          print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}, Qloss: {4}, Disloss: {5}, Conloss: {6}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy(), Q_loss.data.cpu().numpy() , dis_loss.data.cpu().numpy() , con_loss.data.cpu().numpy()  )
          )

          noise.data.copy_(fix_noise)
          dis_c.data.copy_(torch.Tensor(one_hot))

          con_c.data.copy_(torch.from_numpy(c1))
          z = torch.cat([noise, dis_c, con_c], 1).view(-1,self.opt.nnoise+self.opt.ndiscrete+self.opt.ncontinuous, 1, 1)
          x_save = self.G(z)
          save_image(x_save.data,self.opt.outf + '/c1_'+str(epoch)+'.png', nrow=self.opt.ndiscrete)

          con_c.data.copy_(torch.from_numpy(c2))
          z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.opt.nnoise+self.opt.ndiscrete+self.opt.ncontinuous, 1, 1)
          x_save = self.G(z)
          save_image(x_save.data, self.opt.outf + './c2_'+str(epoch)+'.png', nrow=self.opt.ndiscrete)

      torch.save(self.G.state_dict(), '%s/netG_epoch_%d.pth' % (self.opt.outf, epoch))
