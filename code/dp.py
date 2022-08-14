import torch
import time
import numpy as np
from utils import glo
import torch.nn as nn
import random
from utils import index_add



def cost(a,l,r):
    l,r=l-1,r-1
    c=torch.zeros(a.size(1)).to(a.device)
    cnt=0
    for i in range(l,r+1):
        c+=a[i]
        cnt+=1
    c/=cnt
    cost=0
    for i in range(l,r+1):
        cost+=(((a[i]-c)**2).sum())**0.5
    return cost


def dp(a,N):
    n=a.size(0)
    f=[[-1 for _ in range(N+1)] for _ in range(n+1)]
    fr=[[-1 for _ in range(N+1)] for _ in range(n+1)]
    ccost=[[0 for _ in range(n+1)] for _ in range(n+1)]
    for i in range(1,n+1):
        print(i)
        for j in range(i,n+1):
            ccost[i][j]=cost(a,i,j)

    f[0][0]=0
    a.to('cpu')
    for i in range(0,n):
        print(i)
        for j in range(0,N):
            if(f[i][j]==-1):
                continue
            for k in range(i+1,n+1):
                t=f[i][j]+ccost[i+1][k]
                if(f[k][j+1]==-1 or f[k][j+1]>t):
                    f[k][j+1]=t
                    fr[k][j+1]=i
    
    ed=n
    i=N
    belong=[0 for _ in range(n)]
    cnt=[0 for _ in range(N)]
    while(1):
        if(ed==0):
            break
        st=fr[ed][i]+1
        for j in range(st,ed+1):
            belong[j-1]=i-1
        cnt[i-1]=ed-st+1
        ed=st-1
        i-=1

    print(belong)
    print(cnt)
    time.sleep(100)

    return torch.LongTensor(belong),torch.tensor(cnt)


def dp_grouping(a,N):
    device=a.device
    b,h,n,d=a.size()

    cnt=torch.zeros(b,h,N).to(device)
    belong=torch.zeros(b,h,n).to(device)

    for i in range(b):
        for j in range(h):
            belong[i,j],cnt[i,j]=dp(a[i,j],N)

    belong=belong.long()
    group=index_add(a,belong,cnt,N)
    return group,belong,cnt