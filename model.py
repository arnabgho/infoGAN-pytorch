import torch.nn as nn
from common_net import *

class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self,opt):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(opt.nc, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1024, 1, 1),
      nn.Sigmoid()
    )


  def forward(self, x):
    output = self.main(x).view(-1, 1)
    return output


class Q(nn.Module):

  def __init__(self,opt):
    super(Q, self).__init__()
    self.opt = opt
    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, opt.ndiscrete  , 1)
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)

  def forward(self, x):

    y = self.conv(x)
    disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var


class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x)
    return output

class GatedConvResnetG(nn.Module):
    def __init__(self,opt):
        super(GatedConvResnetG, self).__init__()
        self.opt=opt
        self.main_initial = nn.Sequential(
          nn.ConvTranspose2d(opt.nnoise, 4*opt.ngf, 1, 1, bias=False),
          nn.BatchNorm2d(4*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(4*opt.ngf, 2*opt.ngf, 7, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, 2*opt.ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, opt.ngf, 4, 2, 1, bias=False),
          #nn.Sigmoid()
        )

        main_block=[]
        #Input is z going to series of rsidual blocks

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout)] #[BATCHResBlock(opt.ngf,opt.dropout)]


        # Final layer to map to 1 channel

        main_block+=[nn.Conv2d(opt.ngf,1,kernel_size=3,stride=1,padding=1)]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ nn.Linear(opt.nsalient,opt.ngf_gate)]
        #gate_block+=[ nn.BatchNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        gate_block+=[ nn.Linear(opt.ngf_gate,opt.ngf_gate) ]
        #gate_block+=[ nn.BatchNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        gate_block+=[ nn.Linear(opt.ngf_gate,opt.ngres) ]
        gate_block+= [nn.Sigmoid()] #[ nn.Softmax()]# [nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate=nn.Sequential(*gate_block)

    def forward(self, input):
        input = input.view(-1,self.opt.nz)
        input_gate = input[:,self.opt.nnoise:self.opt.nz]
        input_main = input[:,0:self.opt.nnoise]
        input_main = input_main.resize(self.opt.batchSize,self.opt.nnoise,1,1)

        output_gate = self.gate(input_gate)
        output = self.main_initial(input_main)
        for i in range(self.opt.ngres):
            alpha = output_gate[:,i]
            alpha = alpha.resize(self.opt.batchSize,1,1,1)
            output=self.main[i](output,alpha)

        output=self.main[self.opt.ngres](output)
        output=self.main[self.opt.ngres+1](output)
        return output

class GatedResnetConvResnetG(nn.Module):
    def __init__(self,opt):
        super(GatedResnetConvResnetG, self).__init__()
        self.opt=opt
        self.main_initial = nn.Sequential(
          nn.ConvTranspose2d(opt.nnoise, 4*opt.ngf, 1, 1, bias=False),
          nn.BatchNorm2d(4*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(4*opt.ngf, 2*opt.ngf, 7, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, 2*opt.ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, opt.ngf, 4, 2, 1, bias=False),
          #nn.Sigmoid()
        )

        main_block=[]
        #Input is z going to series of rsidual blocks

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout)] #[BATCHResBlock(opt.ngf,opt.dropout)]


        # Final layer to map to 1 channel

        main_block+=[nn.Conv2d(opt.ngf,opt.nc,kernel_size=3,stride=1,padding=1)]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ nn.Linear(opt.nsalient ,opt.ngf_gate)]
        #gate_block+=[ nn.BatchNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        for i in range(opt.ngres_gate):
            gate_block+=[ResBlock(opt.ngf_gate,opt.dropout)]
        gate_block+=[ nn.Linear(opt.ngf_gate,opt.ngres) ]
        gate_block+= [ nn.Sigmoid()]# [nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate=nn.Sequential(*gate_block)

    def forward(self, input):
        input = input.view(-1,self.opt.nz)
        input_gate = input[:,self.opt.nnoise:self.opt.nz]
        input_main = input[:,0:self.opt.nnoise]
        input_main = input_main.resize(self.opt.batchSize,self.opt.nnoise,1,1)

        output_gate = self.gate(input_gate)
        output = self.main_initial(input_main)
        for i in range(self.opt.ngres):
            alpha = output_gate[:,i]
            alpha = alpha.resize(self.opt.batchSize,1,1,1)
            output=self.main[i](output,alpha)

        output=self.main[self.opt.ngres](output)
        output=self.main[self.opt.ngres+1](output)
        return output

class ConvResnetG(nn.Module):
    def __init__(self,opt):
        super(ConvResnetG, self).__init__()
        self.opt=opt
        self.main_initial = nn.Sequential(
          nn.ConvTranspose2d(opt.nz, 4*opt.ngf, 1, 1, bias=False),
          nn.BatchNorm2d(4*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(4*opt.ngf, 2*opt.ngf, 7, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, 2*opt.ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, opt.ngf, 4, 2, 1, bias=False),
          #nn.Sigmoid()
        )

        main_block=[]
        #Input is z going to series of rsidual blocks

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+= [ConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout)] #[BATCHResBlock(opt.ngf,opt.dropout)]


        # Final layer to map to 1 channel

        main_block+=[nn.Conv2d(opt.ngf,1,kernel_size=3,stride=1,padding=1)]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)


    def forward(self, input):
        output=self.main_initial(input)
        output=self.main(output)
        return output.view(-1,1,28,28)


class ResnetFrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self,opt):
        super(ResnetFrontEnd, self).__init__()

        main_block=[]

        #Input is 1D going to series of residual blocks

        main_block+=[nn.Linear(opt.out_dim,opt.ngf) ]
        main_block+=[nn.ReLU()]
        # Sets of residual blocks start

        for i in range(opt.ndres):
            main_block+= [ResBlock(opt.ngf,opt.dropout)] # [BATCHResBlock(opt.ngf,opt.dropout)]

        # Final layer to map to sigmoid output

        #main_block+=[nn.Linear(opt.ngf,1)]
        #main_block+=[nn.Sigmoid()]

        self.main=nn.Sequential(*main_block)


    def forward(self, input):
        input=input.view(-1,784)
        output = self.main(input)
        return output



class ResnetD(nn.Module):

    def __init__(self,opt):
        super(ResnetD, self).__init__()

        main_block = []
        main_block+=[nn.Linear(opt.ngf,1)]
        main_block+=[nn.Sigmoid()]

        self.main=nn.Sequential(*main_block)

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class ResnetQ(nn.Module):

  def __init__(self,opt):
    super(ResnetQ, self).__init__()

    self.conv = nn.Linear(opt.ndf,opt.ndf)  #nn.Conv2d(1024, 128, 1, bias=False)
    #self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Linear(opt.ndf,10)  #nn.Conv2d(128, 10, 1)
    self.conv_mu =  nn.Linear(opt.ndf,2) # nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Linear(opt.ndf,2) #nn.Conv2d(128, 2, 1)

  def forward(self, x):

    y = self.conv(x)

    disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var


class ResnetG(nn.Module):
    def __init__(self,opt):
        super(ResnetG, self).__init__()
        main_block=[]

        #Input is z going to series of rsidual blocks

        main_block+=[nn.Linear(opt.nz,opt.ngf) ]

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+= [ResBlock(opt.ngf,opt.dropout)] #[BATCHResBlock(opt.ngf,opt.dropout)]

        # Final layer to map to 1D

        main_block+=[nn.Linear(opt.ngf,opt.out_dim)]

        self.main=nn.Sequential(*main_block)


    def forward(self, input):
        input = input.view(-1,74)
        output = self.main(input)
        return output.view(-1,1,28,28)

class GatedResnetG(nn.Module):
    def __init__(self,opt):
        super(GatedResnetG, self).__init__()
        main_block=[]
        self.opt=opt
        #Input is z going to series of rsidual blocks

        main_block+=[nn.Linear(opt.nnoise,opt.ngf) ]

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+= [GatedResBlock(opt.ngf,opt.dropout)] #[BATCHResBlock(opt.ngf,opt.dropout)]

        # Final layer to map to 1D

        main_block+=[nn.Linear(opt.ngf,opt.out_dim)]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ nn.Linear(opt.nsalient,opt.ngf)]
        gate_block+=[ nn.ReLU()]
        gate_block+=[ nn.Linear(opt.ngf,opt.ngf) ]
        gate_block+=[ nn.ReLU()]
        gate_block+=[ nn.Linear(opt.ngf,opt.ngres) ]
        gate_block+=[nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate=nn.Sequential(*gate_block)

    def forward(self, input):
        input = input.view(-1,self.opt.nz)
        input_gate = input[:,self.opt.nnoise:self.opt.nz]
        input_main = input[:,0:self.opt.nnoise]

        output_gate = self.gate(input_gate)
        output = self.main[0](input_main)
        for i in range(self.opt.ngres):
            alpha = output_gate[:,i]
            alpha = alpha.resize(self.opt.batchSize,1)
            output=self.main[i+1](output,alpha)

        output=self.main[self.opt.ngres+1](output)
        output=self.main[self.opt.ngres+2](output)
        return output.view(-1,1,28,28)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
