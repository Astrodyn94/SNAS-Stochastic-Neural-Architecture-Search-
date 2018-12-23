import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
  ## Formula (2) in SNAS paper
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, Z):  
    return sum(z * op(x) for z, op in zip(Z, self._ops)) 



class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps): ##_steps is set to 4, which is number of the intermediate nodes
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, Z):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, Z[offset+j]) for j, h in enumerate(states)) ##Mixedop(C,stride)(x,weights)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1) 



class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network,self).__init__()
        self._C = C ## initial number of channels (given)
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C  ## output channel = stem_multiplier *input channel 
        self.stem = nn.Sequential(
        nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        nn.BatchNorm2d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  ## where C equals to the initial channel that is given. 
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]: ## layer is 8 , total number of the cell 
                C_curr *= 2
                reduction = True
            else:
                reduction = False  ##   1/3 지점과 2/3 지점에 reduction cell 만들어준다 
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr ##multipier is given 

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()  ## alpha initialization 
        
    def forward(self, input , temperature):
        s0 = s1 = self.stem(input) ## Initialization of states 
        for i, cell in enumerate(self.cells):   ## cells는 8번 stack한 것이다. 
            if cell.reduction:
                Z, score_function = self.ArchitectDist(self.alphas_reduce , temperature) # shape = [14,8]
            else:
                Z, score_function = self.ArchitectDist(self.alphas_normal , temperature)
            s0, s1 = s1, cell(s0, s1, Z) ## output cell하나 만드는데 이전 2개의 cell들이 필요하다. 
        out = self.global_pooling(s1) 
        logits = self.classifier(out.view(out.size(0),-1))
        return logits , score_function

    def _loss(self, input, target , temperature):
        logits,_ = self(input , temperature ) 
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    '''
    def ArchitectDist(self,alpha):

        m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            torch.tensor([2.2]) , alpha) ###hyperparameter softmax temperature lambda was not given in the papaer... thus used 2.2
        return m.sample() , -m.log_prob(m.sample())
    '''

    def ArchitectDist(self,alpha,temperature):

        m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            torch.tensor([temperature]).cuda() , alpha) ###hyperparameter softmax temperature lambda was not given in the papaer... thus used 2.2
        return m.sample() , -m.log_prob(m.sample())

    def _loss(self, input,target,temperature):
        logits,_ = self(input,temperature)
        return self._criterion(logits, target) 

    ## targets and inputs should be given through DataSet 
    def Credit(self,input,target,temperature):
        loss = self._loss(input,target,temperature)
        dL = torch.autograd.grad(loss,input)[0]
        dL_dX = dL.view(-1); X = input.view(-1)
        credit = torch.dot(dL_dX.double() , X.double())
        #credit = torch.autograd.grad(loss,input)[0] * input
        return credit
