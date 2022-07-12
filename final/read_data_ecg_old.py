from tqdm import tqdm
import random
import scipy.io as sio
import numpy as np
import os
import numpy as np
import time
import csv

from utils import glo

downrate=2
sample_rate=int(500/downrate)
sec_window=None

def read_data_ecg(opt):
    global sec_window,sample_rate
    sec_window=opt.seg_len/sample_rate
    glo.n=sample_rate*sec_window

    p='../../raw_dataset/ECG/'
    
    f = csv.reader(open(p+'/REFERENCE.csv','r'))
    labels=[]
    for l in f:
        if(len(labels)==0):
            labels.append(None)
        else:
            labels.append(int(l[1])-1)

    trials=[]
    total=6878
    if(opt.ins_len!=None):
        total=100
    for i in tqdm(range(1,total//1)):
        label=labels[i]

        pre,suf,name=p+'TrainingSet/A','.mat',str(i)
        while(len(name)<4):
            name='0'+name
        path=pre+name+suf
        t = sio.loadmat(path)['ECG'][0][0][2]
        t=t.transpose()
        for j in range(downrate):
            tt=t[j::downrate,:].tolist()
            trials.append({'data':tt,'label':label})

    ret=trial_split(opt,trials)
    return ret

def trial_split(opt,trials):
    args=opt
    print('trial')
    data={'pretrain':[],'train':[],'dev':[]}

    trials=shuffle_list(trials)
    
    for now in range(opt.num_class):
        cnt=0
        for trial in trials:
            if(trial['label']==now):
                cnt+=1
        id=0
        for trial in trials:
            if(trial['label']==now):
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

    seg_len=int(sec_window*sample_rate)
    ret=[]
    for ins in data:
        st=0
        ds=int(seg_len-seg_len*overlap)
        n=len(ins['data'])

        while(st+seg_len<=n):
            data,lab=ins['data'][st:st+seg_len],ins['label']
            ret.append({'data':data,'label':lab})
            st+=ds
    ret=shuffle_list(ret)
    return ret