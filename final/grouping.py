from re import S
import scipy as sp
import torch
import time
import numpy as np
from utils import glo, speed,tmp
import torch.nn as nn
import random
from dp import dp_grouping
from serial import serial_grouping,threshold_grouping,equal_grouping
from sklearn.cluster import KMeans
from sklearn import preprocessing
from utils import index_add,get_cnt,average,discretize,kmeans_init
from utils import group_distance,kmeans_distance

def dot_grouping(a,m=5):
    device=a.device
    b,h,n,d=a.size()
    N=2**m

def L2(a,b):
    return (((a-b)**2).sum())**(0.5)

def group_check(a,belong,N):
    b,h,n,d=a.size()
    print(belong.size())
    print(belong[0,0])
    for i in range(N):
        ct=-1
        print()
        for j in range(n):
            if(belong[0,0,j]==i):
                ct=j
                break
        if(ct!=-1):
            print(i)
            dis=[]
            for j in range(n):
                if(belong[0,0,j]==i):
                    dis.append(L2(a[0,0,ct],a[0,0,j]))
            dis=torch.tensor(dis)
            dis=torch.sort(dis).values
            print(dis)
        time.sleep(1)


def hash_grouping(a,m=6):
    device=a.device
    b,h,n,d=a.size()

    r=0.05
    R=torch.randn(d,device=device)
    bias=torch.rand(1,device=device)[0]*r
    x=torch.floor(((a*R).sum(-1)+bias)/r)
    min=torch.min(x,-1).values
    x=x-min.unsqueeze(2)
   
    #scale
    '''data=x.view(b*h,n).permute(1,0)
    data=data.cpu().numpy()

    scaler=preprocessing.MinMaxScaler((0,N-1e-6))
    scaler.fit(data)
    data=scaler.transform(data)
    x=torch.from_numpy(data).to(device)
    x=x.permute(1,0).view(b,h,n)
    x=torch.floor(x)''' 

    belong=discretize(x)
    belong=belong.int() if(glo.use_mask) else belong.long()
    N=belong.max().int()

    group_check(a,belong,N)
    '''T=torch.ones(m,d,device=device)
    for i in range(m):
        idx=torch.randperm(d)[:d//2]
        print(idx)
        T[i][idx]=-1
    T=T.permute(1,0)
    T=torch.bernoulli(torch.full(torch.Size([d,m]),0.5,device=device))
    T=T*2-1
    S=torch.sign(torch.matmul(a,T)).int()
    S=(S+1)/2

    pow=torch.zeros(m,1,device=device)
    for i in range(m):
        pow[i,0]=2**i
    
    B=torch.matmul(S,pow)
    
    belong=B.squeeze(-1).int()'''


    myrange=torch.arange(end=N,device=device)
    mask=(belong.unsqueeze (-1) == (myrange)).float()

    cnt=get_cnt(belong,mask,N)
    
    sb=(cnt>0).int()
    print(sb.sum().cpu()/(b*h*N))

    return belong,cnt,mask



def kmeans_grouping(a,N,weight):
    b,h,n,d=a.size()
    compatness=None
    parallel=True
    if(parallel==False):
        belong,cnt,mask=kmeans_cpu(a,N)
    else:
        belong,cnt,mask,compatness=kmeans_gpu(a,N,glo.K_r,weight)

    sb=(cnt>0).int()
    
    return belong,cnt,mask,compatness

def kmeans_gpu(x, N, niter,weight):
    b,h,n, d = x.size()
    device=x.device

    torch.cuda.synchronize()
    st=time.time()
    c=None
    torch.cuda.synchronize()
    speed.grp_zeros+=time.time()-st

    torch.cuda.synchronize()
    st=time.time()
    c=kmeans_init(x,N,init='rand')
    torch.cuda.synchronize()
    speed.grp_ini+=time.time()-st
    
    cnt,compatness_sum=None,None

    for i in range(niter):
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的N类
        
        if(i<niter-1):
            belong,_=kmeans_distance(x,c)
        else:
            belong,compatness_sum=kmeans_distance(x,c,return_compatness=True)
        
        mask=None
        if(glo.use_mask==True):
            myrange=torch.arange(end=N,device=device)
            mask=(belong.unsqueeze (-1) == (myrange)).float()
            '''min=torch.min(belong,3,keepdim=True).values
            mask=torch.exp((min-belong)*10000)'''
        
        torch.cuda.synchronize()
        st=time.time()
        cnt=get_cnt(belong,mask,N,weight)
        torch.cuda.synchronize()
        speed.grp_cnt+=time.time()-st
       
        if(i<niter-1):
            torch.cuda.synchronize()
            st=time.time()
            c=average(x,belong,cnt,mask,average=True,weight=weight)
            torch.cuda.synchronize()
            speed.grp_upd+=time.time()-st
    
    #group_check(x,belong,N)
    
    compatness=compatness_sum/cnt
    return belong,cnt,mask,compatness

    
def kmeans_cpu(a,N):
    device=a.device
    b,h,n,d=a.size()
    belong=torch.zeros(b,h,n,device=device).long()
    torch.cuda.synchronize()
    st=time.time()
    for i in range(b):
        for j in range(h):
            belong[i,j]=kmeans_cpu(a[i,j],N)
            c = a[torch.randperm(n)[:N]].cpu().numpy()
            data=a.cpu().numpy()
            
            cluster=KMeans(n_clusters=N,n_init=1,max_iter=1,init=c)
            cluster.fit(data)
            belong[i,j]=torch.LongTensor(cluster.labels_,device=device)
    torch.cuda.synchronize()
    speed.grp_sklearn+=time.time()-st

    cnt=get_cnt(...)

    return belong,cnt,None
    



def grouping(a,scheme='kmeans',N=None,weight=None):
    with torch.no_grad():
        if(scheme=='hash'):
            return hash_grouping(a)
        elif(scheme=='serial'):
            return serial_grouping(a)
        elif(scheme=='dot'):
            return dot_grouping(a)
        elif(scheme=='equal'):
            return equal_grouping(a)
        elif(scheme=='kmeans'):
            return kmeans_grouping(a,N,weight)
        elif(scheme=='dp'):
            return dp_grouping(a,N)
        elif(scheme=='threshold'):
            return threshold_grouping(a)