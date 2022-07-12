from pretrain import pretrain
from train import train
from dev import dev

from raw_to_input import raw_to_input
from utils import set_all_seed,set_data_seed,glo,fetch_few_data
from N_schedule import N_schedule
import random
from dataset import Dataset,make_dataset,read_dataset,dataset_dist

from TST import model_factory
from simpleLM import SIMPLE
from myLM import BERT

from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json
from read_data_hhar import read_data_hhar
from read_data_rwhar import read_data_rwhar
from read_data_wisdm import read_data_wisdm
from read_data_ecg_old import read_data_ecg
from read_data_stock import read_data_stock
from read_data_mgh import read_data_mgh


def dist(data,args):
    t=[0 for _ in range(args.num_class)]

    batch_size=1024
    st=0
    while(st<data.len):
        ed=min(data.len,st+batch_size)
        n=ed-st
        batch=data.fetch(st,ed,args)
        for i in range(n):
            lab=int(batch['label'][i])
            t[lab]+=1
        st=ed

    cnt=0
    for i in range(args.num_class):
        cnt+=t[i]
    
    print(t,cnt)

def ddist(dataset_pretrain,dataset_train,dataset_dev,args):
    if(args.task=='for' or args.task=='imp'):
        return
    dist(dataset_pretrain,args)
    dist(dataset_train,args)
    dist(dataset_dev,args)
    print()



def main(args):
    device=args.device

    print(args.task,'on',args.dataset+str(args.seg_len))

    print('creating model: seed',args.all_seed)
    set_all_seed(args)
    if(args.model=='BERT'):
        model=BERT(hidden_siz=args.hidden_siz,num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,intermediate_siz=args.intermediate_size,
        num_class=args.num_class,device=device,args=args).to(device)
    elif(args.model=='TST'):
        model=model_factory(args).to(device)
    else:
        model=SIMPLE()

    print('loading data: seed',args.data_seed)
    set_data_seed(args)
    if(not args.data_method in ['semi','trial','sample']):
        raise ValueError('data method error')
    
    data={'train':None,'dev':None}
    dataset_name=args.dataset+str(args.seg_len)+'_'+str(args.pre)+'-'+str(args.train)+'-'+str(args.dev)
    tensor_path='../../tensor_dataset/'+dataset_name
    import os
    if(os.path.exists(tensor_path)==False):
        print('read dataset')
        if(args.dataset=='wisdm'):
            data=read_data_wisdm(args)
        elif(args.dataset=='rwhar'):
            data=read_data_rwhar(args)
        elif(args.dataset=='hhar'):
            data=read_data_hhar(args)
        elif(args.dataset=='ecg'):
            data=read_data_ecg(args)
        elif(args.dataset=='stock'):
            data=read_data_stock(args)
        elif(args.dataset=='mgh'):
            data=read_data_mgh(args)
        data=raw_to_input(args,data)
        

    glo.N=[glo.n for _ in range(args.num_layers)]
    if(args.N_init==None):
        args.N_init=glo.n
    
    if(args.mode=='finetune'):
        if(args.restore_model=='None'):
            args.restore_model='../pretrained/'+args.dataset+str(args.seg_len)+'_0.9-0.0-0.1'+args.model_name

    print('begin:',args.mode,'seed',args.all_seed)
    set_all_seed(args)
    writer= SummaryWriter(args.log_name)

    glo.args=args

    if(os.path.exists(tensor_path)==False):
        os.mkdir(dataset_name)
        dataset_pretrain=make_dataset(data['pretrain'],'pretrain',args,dataset_name)
        dataset_train=make_dataset(data['train'],'train',args,dataset_name)
        dataset_dev=make_dataset(data['dev'],'dev',args,dataset_name)
        dataset_dist(dataset_name)
    else:
        dataset_pretrain=read_dataset(tensor_path,'pretrain',args)
        dataset_train=read_dataset(tensor_path,'train',args)
        dataset_dev=read_dataset(tensor_path,'dev',args)
        dataset_dist(tensor_path)

    if(args.ins_len!=None):
        num=3 if(args.device=='cpu') else 1200
        args.batch_size=1 if(args.device=='cpu') else 8

        data_train=[]
        for _ in range(num):
            t=[]
            for i in range(args.ins_len):
                t.append([random.random() for _ in range(args.num_channel)])
            data_train.append({'input':t,'label':0})
        data['train']=data_train

    if(args.few!=None):
        dataset_train=fetch_few_data(dataset_train,args)
    ddist(dataset_pretrain,dataset_train,dataset_dev,args)
    
    if(args.mode=='pretrain'):
        pretrain(dataset_pretrain,dataset_dev,args,model,writer,device)
    elif(args.mode=='train' or args.mode=='finetune'):
        train(dataset_train,dataset_dev,args,model,writer,device)
    elif(args.mode=='dev'):
        dev(dataset_dev,args,model,writer,device)
    elif(args.mode=='N_schedule'):
        N_schedule(args,device)