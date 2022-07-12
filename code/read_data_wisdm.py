label_set=['A','B','C','D','E','F','G','H','I','J','K','L','M','O','P','Q','R','S']


from tqdm import tqdm
import random

def read_data_wisdm(opt):
    print('loading data',opt.data_method,opt.pre_overlap,opt.data_overlap)
    opt.datafile='accel_watch'
    dir='../../scaled_dataset/wisdm/'+opt.datafile+'/'
    trials=[]

    for i in tqdm(range(51)):
        path=dir+'data_'+str(1600+i)+'_'+opt.datafile+'.txt'
        f=open(path,'r')
        data=[]

        plable=None
        while(1):
            s=f.readline()
            if(s==''):
                break
            s=s[:-2]
            s=s.split(',')
            label=s[1]
            if(label!=plable):
                if(label in label_set):
                    data.append({'label':label_set.index(label),'x':[],'y':[],
                                'z':[]})

            x,y,z=float(s[3]),float(s[4]),float(s[5])
            n=len(data)-1
            data[n]['x'].append(x)
            data[n]['y'].append(y)
            data[n]['z'].append(z)
            plable=label
        
        for trial in data:
            trials.append(trial)
    
    ret=None
    if(opt.data_method=='semi'):
        ret=semi_split(opt,trials)
    elif(opt.data_method=='trial'):
        ret=trial_split(opt,trials)
    elif(opt.data_method=='sample'):
        ret=sample_split(opt,trials)

    return ret

def sample_split(opt,trials):
    new_trials=[]
    print('sample')

    for trial in trials:
        dlt=30*20

        label,x,y,z=trial['label'],trial['x'],trial['y'],trial['z']
        st=0
        n=len(x)
        while(st<n):
            ed=st+dlt
            if(ed>n):
                ed=n
            new_trials.append({'label':label,'x':x[st:ed],'y':y[st:ed],'z':z[st:ed]})
            st=ed
    return trial_split(opt,new_trials)



def semi_split(opt,trials):
    print('semi')

    data=trials_to_ins(opt,trials,'train')

    ret={'pretrain':[],'train':[],'dev':[]}
    cnt=len(data)
    for i in range(len(data)):
        if(i<cnt*opt.pre):
            ret['pretrain'].append(data[i])
        elif(i<=cnt*(opt.pre+opt.train)):
            ret['train'].append(data[i])
        else:
            ret['dev'].append(data[i])

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
    overlap=None
    if(ty=='pretrain'):
        overlap=opt.pre_overlap
    elif(ty=='train' or ty=='dev'):
        overlap=opt.data_overlap

    ret=[]
    for ins in data:
        st=0
        seg_len=opt.seg_len
        ds=int(seg_len-seg_len*overlap)

        while(st+seg_len<=len(ins['x'])):
            x,y=ins['x'][st:st+seg_len],ins['y'][st:st+seg_len]
            z=ins['z'][st:st+seg_len]
            lab=ins['label']
            ret.append({'x':x,'y':y,'z':z,'label':lab})
            st+=ds
    ret=shuffle_list(ret)

    return ret