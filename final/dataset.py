import torch
import numpy as np
import time
import json
import os

class Dataset:
    def __init__(self,args) -> None:
        self.now=0
        self.len=0
        self.st,self.ed=[],[]
        self.name=None
        self.path=''
        self.imp_method=args.imp_method
        self.mask_rate=args.mask_rate
        

    def start(self,args):
        self.now=0
        self.load_block(self.now,args)
        

    def fetch(self,l,r,args):
        batch=None
        for i in range(len(self.st)):
            ll,rr=self.intersec(self.st[i],self.ed[i],l,r)
            if(ll!=None):
                tmp=self.fetch_once(i,ll,rr,args)
                batch=self.concat(batch,tmp)
        return batch
    
    def fetch_once(self,now,l,r,args):
        if(self.now!=now):
            self.now=now
            self.load_block(now,args)
        dl,dr=l-self.st[now],r-self.st[now]
        return self.get_data(dl,dr)
    

    def load_block(self,now,args):
        input,label,empty=load_data_block(now,self.name,self.path)
        if(empty==True):
            return
        
        if(args.task=='imp'):
            orig=torch.from_numpy(input)

            if(args.cut_len!=None):
                orig=orig[:,:args.cut_len]
            
            b,n,d=orig.size()
            

            mask,replace=random_mask(orig,self.imp_method,self.mask_rate)

            self.input=((1-replace).unsqueeze(-1)*orig-replace.unsqueeze(-1)).detach().clone()
            self.label=orig.detach().clone()
            self.mask=mask.detach().clone()
        elif(args.task=='cls'):
            input=torch.from_numpy(input)
            label=torch.from_numpy(label)
            b,_,_=input.size()

            self.input=input.detach().clone()
            self.label=label.detach().clone()
            self.mask=torch.zeros(b)


            
    def intersec(self,l,r,ll,rr):
        if(l>ll):
            l,r,ll,rr=ll,rr,l,r
        t=min(rr,r)
        if(r<=ll or t==ll):
            return None,None
        return ll,t
    
    def concat(self,batch1,batch2):
        if(batch1!=None):
            batch={'input':torch.cat([batch1['input'],batch2['input']],0),
               'label':torch.cat([batch1['label'],batch2['label']],0),
               'mask' :torch.cat([batch1['mask'],batch2['mask']],0)}
        else:
            batch={'input':batch2['input'],
               'label':batch2['label'],'mask' :batch2['mask']}
        return batch
    
    def get_data(self,l,r):
        batch={'input':self.input[l:r].detach().clone().float(),
               'label':self.label[l:r].detach().clone().float(),
               'mask':self.mask[l:r].detach().clone().float()}
        return batch


def dataset_dist(path):
    for name in ['pretrain','train','dev']:
        t=load_data_config(path,name)
        print(name,t['len'])



def make_dataset(data,name,args,path='./'):
    print('making dataset in',path)
    dataset=Dataset(args)

    dataset.len=len(data)
    dataset.name=name
    dataset.path=path
    block_size=5000*10000//args.seg_len

    st=0
    num=0
    while(st<len(data)):
        ed=min(len(data),st+block_size)
        
        input,label=[],[]
        for i in range(st,ed):
            input.append(data[i]['input'])
            label.append(data[i]['label'])
        
        save_data_block(input,label,num,name,path)
        
        dataset.st.append(st)
        dataset.ed.append(ed)
        st=ed
        num+=1
    
    save_data_config(dataset,name,path)
    
    dataset.start(args)
    return dataset



def read_dataset(path,name,args):
    dataset=Dataset(args)
    print('reading dataset in',path)
    
    t=load_data_config(path,name)
    dataset.path=path
    dataset.len,dataset.st,dataset.ed=t['len'],t['st'],t['ed']
    dataset.name=name
    dataset.start(args)
    return dataset




def make_dataset_from_tensor(input,label,mask,args):
    device=args.device
    dataset=Dataset(args)
    dataset.len,dataset.st,dataset.ed=len(input),[0],[len(input)]
    dataset.now=0
    dataset.input=torch.tensor(input,device=device)
    dataset.label=torch.tensor(label,device=device)
    dataset.mask=torch.tensor(mask,device=device)
    return dataset
    











def save_data_config(dataset,name,path):
    t={'len':dataset.len,'st':dataset.st,'ed':dataset.ed,'name':dataset.name}
    with open(path+'/'+name+'_config','w') as f:
        f.write(json.dumps(t))

def load_data_config(path,name):
    with open(path+'/'+name+'_config','r') as f:
        t=json.loads(f.readline())
    return t















def save_data_block(input,label,num,name,path):
    print('saving block',num)
    input,label=np.array(input),np.array(label)
    input_path=path+'/'+name+'_input_'+str(num)
    label_path=path+'/'+name+'_label_'+str(num)
    np.save(input_path,input,)
    np.save(label_path,label)


def load_data_block(num,name,path):
    print('loading block',num)
    input_path=path+'/'+name+'_input_'+str(num)+'.npy'
    label_path=path+'/'+name+'_label_'+str(num)+'.npy'
    empty=None
    try:
        input=np.load(input_path,allow_pickle=True)
        label=np.load(label_path,allow_pickle=True)
        empty=False
    except:
        input,label=None,None
        empty=True
    
    return input,label,empty
















def random_mask(orig,stra,mask_rate):
    b,n,d=orig.size()
    device=orig.device
    mask,replace=None,None
    dlt=0.05
    major_rate=mask_rate+dlt*4
    minor_rate=mask_rate-dlt
    if(stra=='rand'):
        rand1=torch.rand(b,n)
        rand2=torch.rand(b,n)

        mask=(rand1<mask_rate).int()
        replace=(mask*(rand2<0.9).int())
    elif(stra=='var'):
        seg=n//10
        print(seg)
        x=orig.transpose(-1,-2)
        x2=torch.square(x)
        x_pre=torch.cumsum(x,-1)
        x2_pre=torch.cumsum(x2,-1)

        e_x2=(x2_pre[:,:,seg:]-x2_pre[:,:,:-seg])/seg
        ex_2=torch.square((x_pre[:,:,seg:]-x_pre[:,:,:-seg])/seg)

        var=(e_x2-ex_2).sum(1)

        weight=torch.zeros(b,n,device=device)
        weight[:,seg//2:seg//2+(n-seg)]=var


        thre=torch.quantile(weight,0.8,-1).unsqueeze(-1)
        minor=(weight<thre).int()
        prob=minor*minor_rate+(1-minor)*major_rate
        
        mask=torch.bernoulli(prob)
        rand2=torch.rand(b,n)
        replace=(mask*(rand2<0.9).int())

    return mask,replace


if __name__=='__main__':
    b,n,d=2,100,4
    a=torch.randn(b,n,d)
    mask,replace=random_mask(a,'var')
