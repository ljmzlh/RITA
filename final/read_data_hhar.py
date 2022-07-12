
from tqdm import tqdm
import random
import json

mapping={'null':-1,'walk':0,'bike':1,'stairsup':2,'stairsdown':3,'sit':4,'stand':5}

def read_data_hhar(opt):
    print('loading data',opt.pre_overlap,opt.data_overlap)

    path='../../scaled_dataset/hhar/'+opt.datafile

    f=open(path,'r')
    trails=[]
    for line in f:
        t=json.loads(line)
        t['label']=mapping[t['label']]
        assert t['label']>=-1 and t['label']<=5
        trails.append(t)
    f.close()
    
    ret=trial_split(opt,trails)

    return ret

def trial_split(args,trials):
    print('trial')
    data={'pretrain':[],'train':[],'dev':[]}

    trials=shuffle_list(trials)
    
    for now in range(args.num_class):
        cnt=0
        for trial in trials:
            if(trial['label']==now):
                cnt+=1
        
        id=0
        for trial in trials:
            if(trial['label']==now):
                id+=1
                if(id<=args.pre*cnt or trial['label']==-1):
                    data['pretrain'].append(trial)
                elif(id<=(args.pre+args.train)*cnt):
                    data['train'].append(trial)
                else:
                    data['dev'].append(trial)

    ret={}
    ret['pretrain']=trials_to_ins(args,data['pretrain'],'pretrain')
    ret['train']=trials_to_ins(args,data['train'],'train')
    ret['dev']=trials_to_ins(args,data['dev'],'dev')
    return ret


def shuffle_list(a):
    for i in range(1,len(a)):
        j=random.randint(0,i-1)
        a[i],a[j]=a[j],a[i]
    return a

def trials_to_ins(opt,data,ty):
    overlap=None
    if(ty=='pretrain'):
        overlap=opt.pre_overlap
    elif(ty=='train' or ty=='dev'):
        overlap=opt.data_overlap

    ret=[]
    for ins in data:
        st=0
        seg_len=200
        ds=int(seg_len-seg_len*overlap)

        while(st+seg_len<=len(ins['x'])):
            x,y=ins['x'][st:st+seg_len],ins['y'][st:st+seg_len]
            z=ins['z'][st:st+seg_len]
            lab=ins['label']
            ret.append({'x':x,'y':y,'z':z,'label':lab})
            st+=ds
    ret=shuffle_list(ret)

    return ret