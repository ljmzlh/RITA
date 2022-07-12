from tqdm import tqdm
import random
import scipy.io as sio
import numpy as np
import os
import numpy as np
import time
import csv
import json

from utils import glo


def read_data_mgh(opt):

    trials=[]
    total=500
    if(opt.ins_len!=None):
        total=10
    for i in tqdm(range(1,total+1)):
        f=open('/home/ljmzlh/scaled_dataset/mgh/'+str(i),'r')
        a=json.loads(f.readline())
        trials.append(a)

    ret=trial_split(opt,trials)
    return ret

def trial_split(opt,trials):
    args=opt
    print('trial')
    data={'pretrain':[],'train':[],'dev':[]}

    trials=shuffle_list(trials)
    
    cnt=len(trials)
    id=0
    for trial in trials:
            id+=1
            if(id<=opt.pre*cnt):
                    data['pretrain'].append(trial)
            elif(id<=(opt.pre+opt.train)*cnt):
                    data['train'].append(trial)
            else:
                    data['dev'].append(trial)

    ret={}
    ret['pretrain']=trials_to_ins(opt,data['pretrain'],'pretrain')
    ret['train']=trials_to_ins(opt,data['train'],'train')
    if(args.few!=None):
        ret['train']=[]
        cnt=[0 for _ in range(args.num_class)]
        for ins in ret['pretrain']:
            if(cnt[ins['label']]==args.few):
                continue
            ret['train'].append(ins)
            cnt[ins['label']]+=1
    ret['dev']=trials_to_ins(opt,data['dev'],'dev')

    return ret

def shuffle_list(a):
    for i in range(1,len(a)):
        j=random.randint(0,i-1)
        a[i],a[j]=a[j],a[i]
    return a

def trials_to_ins(opt,data,ty):
    global sec_window
    overlap=None
    if(ty=='pretrain'):
        overlap=opt.pre_overlap
    elif(ty=='train' or ty=='dev'):
        overlap=opt.data_overlap

    seg_len=10000
    ret=[]
    for ins in data:
        assert opt.num_channel==len(ins[0])

        st=0
        ds=int(seg_len-seg_len*overlap)
        n=len(ins)

        while(st+seg_len<=n):
            data=ins[st:st+seg_len]
            
            ret.append({'data':data,'label':None})
            st+=ds
    ret=shuffle_list(ret)
    return ret