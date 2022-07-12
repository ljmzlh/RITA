import os
import random
import time
import math

import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import grad
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, dataloader)
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import glo, my_save, output_speed, speed, tmp, my_restore
from utils import update_warming_N, my_linear_schedule

import GPUtil
from threading import Thread
import time
from myLM import BERT
import json

grads = {}

def save_grad(name):
    print('fuck')
    def hook(grad):
        grads[name] = grad
    return hook


res=[[0 for _ in range(18)] for _ in range(18)]




def adjust_batch_size(args):
    glo.batch_size=args.batch_size


def process_bar(percent):
    bar = '\r' + ' {:0>4.1f}%'.format(percent*100) 
    print(bar, end='', flush=True)
    if(percent>0.999):
        print()

def collate(batch):
    inputs,pos_id,ty_id,label=[],[],[],[]
    for ins in batch:
        inputs.append(ins['input'])
        label.append(ins['label'])
    inputs=torch.tensor(inputs)
    label=torch.tensor(label)
    return {'input':inputs,'label':label}


def cal_loss(pred,label,device):
    logits=pred.to(device)
    gt=label.to(device)
    loss=F.cross_entropy(logits,gt,reduction='mean')
    return loss


def train_epoch(data_train,args,model,optimizer,scheduler,device):
    model.train()
    optimizer.zero_grad()
    tot_loss=0
    torch.autograd.set_detect_anomaly(True)

    batch_st=0
    num_batch=0
    mx=0
    while(batch_st<len(data_train)):
        num_batch+=1
        if(num_batch>3):
            break
        adjust_batch_size(args)
        batch_ed=min(len(data_train),batch_st+glo.batch_size)
        batch=collate(data_train[batch_st:batch_ed])

        input,label=batch['input'].to(device),batch['label'].to(device)

        torch.cuda.synchronize()
        stt=time.time()
        pred=model.classify(input=input,args=args)
        loss=cal_loss(pred,label,device)
            
        torch.cuda.synchronize()
        speed.fw+=time.time()-stt

        mx=max(mx,GPUtil.getGPUs()[0].memoryUtil)

        assert args.gradient_accumulation_steps==1
        
        if(args.ins_len==None):
            tot_loss+=loss
        
        torch.cuda.synchronize()
        stt=time.time()
        loss.backward()

        mx=max(mx,GPUtil.getGPUs()[0].memoryUtil)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_st=batch_ed
        process_bar(1.0*batch_ed/len(data_train))
    
    return mx


def bsz_measure(data_train,args,model,optimizer,scheduler,device,last_bsz):
    for i in range(args.num_layers):
        glo.N[i]=args.N_init
    print()
    print(args.N_init)
    l,r=1,last_bsz
    upb=0.95
    lwb=0.9
    ret=1

    while(1):
        print()
        while(1):
            torch.cuda.empty_cache()
            a=GPUtil.getGPUs()[0]
            if(a.memoryUtil<0.1):
                break
        print()
        mid=(l+r)//2
        mx=0
        args.batch_size=mid
        
        try:
            mx=train_epoch(data_train,args,model,optimizer,scheduler,device)
        except:
            mx=101

        
        print(l,r,mid,mx)
        
        if(lwb<=mx and mx<=upb):
            return mid
        elif(mx>upb):
            r=mid-1
        else:
            l=mid+1
            ret=max(ret,mid)

        if(l>r):
            break
    
    return ret
    



def N_schedule(args,device):
    glo.measure_kmeans=False

    global global_sample,last_log,last_dev,last_N

    num=3 if(args.device=='cpu') else 500

    data_train=[]
    for _ in range(num):
        t=[]
        for i in range(args.ins_len):
            t.append([random.random() for _ in range(args.num_channel)])
        data_train.append({'input':t,'label':0})
    args.epoch=1


    N_init=args.N_init
    step=N_init//40
    output={'N':[],'bsz':[]}
    last_bsz=400
    for N in range(step,N_init+1,step):
        model,optimizer,scheduler=init_model(args,data_train)

        args.N_init=N
        bsz=bsz_measure(data_train,args,model,optimizer,scheduler,device,last_bsz)
        output['N'].append(N*args.num_layers)
        output['bsz'].append(bsz)
        last_bsz=bsz
    
    print(output)
    file=open('sb','w')
    file.write(json.dumps(output))




def init_model(args,data_train):
    device='cuda'
    model=BERT(hidden_siz=args.hidden_siz,num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,intermediate_siz=args.intermediate_size,
        num_class=args.num_class,device=device,args=args).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    t_total = len(data_train) // 10 // args.gradient_accumulation_steps * args.epoch
    scheduler = my_linear_schedule(optimizer,t_total)

    return model,optimizer,scheduler