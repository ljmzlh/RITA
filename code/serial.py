import torch
import time
import numpy as np
from utils import glo
import torch.nn as nn
import random
from utils import index_add
from multiprocessing import Pool

class tmp:
    belong=None
    eps=None
    N=0
    p=None

geps=None

def dot_product(a,b):
    return (a*b).sum()

def takeSecond(elem):
    return elem[1]

def takeFirst(elem):
    return elem[0]


def equal_grouping(a,m=5):
    device=a.device
    b,h,n,d=a.size()
    N=2**m
    if(tmp.belong==None):
        tmp.belong=torch.zeros(b,h,n).to(device)
        t=torch.zeros(n).to(device)
        m=(n//N)+1
        for i in range(n):
            t[i]=i//m
        tmp.belong[:,:]=t
        tmp.belong=tmp.belong.int()
        
    group=torch.zeros(b,h,N,d).to(device)
    belong=tmp.belong
    for i in range(b):
        for j in range(h):
            idx=belong[i,j]
            group[i,j].index_add_(0,idx,a[i,j])
    return group,belong.long()


def serial_grouping(a,m=5):
    device=a.device
    b,h,n,d=a.size()
    N=2**m

    group=torch.zeros(b,h,N,d).to(device)
    belong=torch.zeros(b,h,n).to(device)
    cnt=torch.zeros(b,h,N).to(device)
    for i in range(b):
        for j in range(h):
            score=[]
            for k in range(n-1):
                t=dot_product(a[i,j,k],a[i,j,k+1]).detach().cpu().tolist()
                score.append((k,t))
            score.sort(key=takeSecond)
            br=score[:N-1]
            br.sort(key=takeFirst)
            br.append((n-1,-1))
            st=0
            for k in range(len(br)):
                ed=br[k][0]
                cnt[i,j,k]=ed-st+1
                for kk in range(st,ed+1):
                    belong[i,j,kk]=k
                st=ed+1
    belong=belong.long()
    group=index_add(a,belong,cnt,N)
    return group,belong,cnt

  



def sequential(a):
    device=a.device
    b,h,n,d=a.size()

    belong=torch.zeros(b,h,n).to(device)
    N=0
    for i in range(b):
        for j in range(h):
            ct,t=0,1
            for k in range(n):
                d=dis(a[i,j,k],a[i,j,ct])
                if(d>tmp.eps):
                    ct=k
                    t+=1
                belong[i,j,k]=t-1
                
            N=max(N,t)
    cnt=torch.zeros(b,h,N).to(device)
    belong=belong.long()
    for i in range(b):
        for j in range(h):
            for k in range(n):
                cnt[i,j,belong[i,j,k]]+=1
    return belong,cnt,N

def parallel(a):
    device=a.device
    b,h,n,d=a.size()
    aa=[]
    for i in range(b):
        for j in range(h):
            aa.append(a[i,j].detach().cpu())

    with Pool(b*h) as p:
        res=p.map(f,aa)

    tmp.N=0
    belong=[[] for _ in range(b*h)]
    for i in range(b*h):
        belong[i]=res[i][0]
        tmp.N=max(tmp.N,res[i][1])

    with Pool(b*h) as p:
        cnt=p.map(g,belong)

    rbelong,rcnt=torch.zeros(b,h,n).int(),torch.zeros(b,h,tmp.N)
    for i in range(b):
        for j in range(h):
            rbelong[i,j],rcnt[i,j]=torch.IntTensor(belong[i*h+j]),cnt[i*h+j]
    return rbelong,rcnt,tmp.N


def c(a,eps):
    a=a.cpu().detach().numpy().astype('double')
    belong,cnt=cg(a,eps)

    N=cnt.shape[2]
    print(N)
    belong=torch.from_numpy(belong).long()
    cnt=torch.from_numpy(cnt).long()
    return belong,cnt,N



        

def threshold_grouping(a,eps=1):
    device=a.device
    b,h,n,d=a.size()

    #belongs,cnt,N=sequential(aa)

    '''st=time.time()
    belongp,cnt,N=parallel(aa)
    print('parallel',time.time()-st)'''

    #st=time.time()
    belong,cnt,N=c(a,eps)
    #print('c',time.time()-st)

    belong=belong.to(device)
    cnt=cnt.to(device).int()

    glo.tot_n+=n
    glo.tot_N+=N
    
    return belong,cnt,None






def dis(a,b):
    return (((a-b)**2).sum())**0.5

def f(a):
    st=time.time()
    n,d=a.size()
    belong=[0 for _ in range(n)]
    N=0
    ct,t=0,1
    for i in range(n):
        d=dis(a[i],a[ct])
        if(d>tmp.eps):
            ct,t=i,t+1
        belong[i]=t-1
    N=max(N,t)
    return belong,N

def g(belong):
    n=len(belong)
    cnt=torch.zeros(tmp.N)
    for i in range(n):
        cnt[belong[i]]+=1
    return cnt