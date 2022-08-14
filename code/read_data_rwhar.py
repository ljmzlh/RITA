
from tqdm import tqdm
import random
import json

def read_data_rwhar(opt):

    path='../rita_dataset/rwhar/'+opt.datafile

    f=open(path,'r')
    trials=json.loads(f.readline())
    f.close()
    
    ret=sample_split(opt,trials)

    return ret

def sample_split(args,trials):
    new_trials=[]

    for trial in trials:
        dlt=20*50

        label,x,y,z=trial['label'],trial['x'],trial['y'],trial['z']
        st=0
        n=len(x)
        while(st<n):
            ed=st+dlt
            if(ed>n):
                ed=n
            new_trials.append({'label':label,'x':x[st:ed],'y':y[st:ed],'z':z[st:ed]})
            st=ed
    return trial_split(args,new_trials)


def trial_split(args,trials):
    print('Trialing')
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
                if(id<=args.pre*cnt):
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