import numpy as np
import utils
import torch 
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
from model_search import Network
from option.default_option import TrainOptions
import os 
import tqdm
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
opt = TrainOptions()
CIFAR_CLASSES = 10
criterion = nn.CrossEntropyLoss().cuda()
model = Network(opt.init_channels, CIFAR_CLASSES, opt.layers, criterion)
model.cuda()
optimizer_model = torch.optim.SGD(model.parameters(),lr= 0.025,momentum = 0.9, weight_decay=3e-4)
optimizer_arch = torch.optim.Adam(model.arch_parameters(),lr = 3e-4, betas=(0.5, 0.999), weight_decay = 1e-3)

train_transform, valid_transform = utils._data_transforms_cifar10(opt)
train_data = dset.CIFAR10(root='../', train=True, download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))

####DATALOADER 수정 필요한부분 
train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=opt.batch_size,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:5000]),
  pin_memory=True, num_workers=2)

valid_queue = torch.utils.data.DataLoader(
  train_data, batch_size=opt.batch_size,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[5000:10000]),
  pin_memory=True, num_workers=2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_model, float(opt.epochs), eta_min=opt.learning_rate_min)

f = open("Loss.txt","w")

def train(train_queue,valid_queue, model, criterion, optimizer_arch, optimizer_model,lr_arch,lr_model):
    objs = utils.AvgrageMeter()
    policy  = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in tqdm.tqdm(enumerate(train_queue)):
        model.train()
        n = input.size(0) # batch size 

        input = Variable(input , requires_grad = True).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)
        
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search , requires_grad = True ).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)
        
        temperature = opt.initial_temp * np.exp(-opt.anneal_rate * step)

        optimizer_arch.zero_grad()
        optimizer_model.zero_grad()
        logit , _ = model(input , temperature)## model inputs 
        _ , score_function = model(input_search , temperature)## model inputs 
        
        policy_loss = torch.sum(score_function * model.Credit(input_search,target_search,temperature).float())
        value_loss = criterion(logit , target) 
        total_loss = policy_loss + value_loss
        total_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),5)
        optimizer_arch.step()
        optimizer_model.step()

        prec1, prec5 = utils.accuracy(logit, target, topk=(1, 5))
        objs.update(value_loss.data, n)
        policy.update(policy_loss.data , n)
        top1.update(prec1.data , n)
        top5.update(prec5.data , n)
        
    return top1.avg, top5.avg, objs.avg, policy.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in tqdm.tqdm(enumerate(valid_queue)):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    temperature = opt.initial_temp * np.exp(-opt.anneal_rate * step)
    logits,_ = model(input , temperature)
    loss = criterion(logits, target)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data , n)
    top1.update(prec1.data , n)
    top5.update(prec5.data , n)
    #if step % args.report_freq == 0:
    #  logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg ,objs.avg

####MAIN 함수 
val_acc_top5 = []
val_acc_top1 = []
for epoch in range(opt.epochs):
    np.random.seed(2)
    torch.cuda.manual_seed(2)
    # training
    train_acc_top1, train_acc_top5 , train_valoss,train_poloss = train(train_queue, valid_queue, model,criterion, optimizer_arch,optimizer_model,opt.arch_learning_rate,opt.lr_model)

    # validation
    valid_acc_top1,valid_acc_top5, valid_valoss = infer(valid_queue, model, criterion)

    f.write("%5.5f  "% train_acc_top1)
    f.write("%5.5f  "% train_acc_top5)
    f.write("%5.5f  "% train_valoss)
    f.write("%5.5f  "% train_poloss ) 
    f.write("%5.5f  "% valid_acc_top1 ) 
    f.write("%5.5f  "% valid_acc_top5 ) 
    f.write("%5.5f  "% valid_valoss ) 
    f.write("\n")


    print("epoch : " , epoch , "Train_Acc_Top1 : " , train_acc_top1 , "Train_value_loss : ",train_valoss,"Train_policy : " , train_poloss )
    print("epoch : " , epoch, "Val_Acc_Top1 : " , valid_acc_top1 , "Val_Acc_Top5 : " , valid_acc_top5,"Loss : " , valid_valoss)
    torch.save(model.state_dict(),'weights.pt')
f.close()



