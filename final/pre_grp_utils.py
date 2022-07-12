import torch
from utils import glo
import time


def all_distance(a):
    dist=cal_dis(a,a)
    return dist

def cal_dis(a,b):
    ab=torch.matmul(a,b.transpose(-1,-2))
    a2=torch.square(a).sum(-1)
    b2=torch.square(b).sum(-1)
    dist2=a2.unsqueeze(-1)+b2.unsqueeze(-2)-2*ab
    dist=torch.sqrt(torch.abs(dist2))
    return dist

def cal(belong,a):
    b,n,d=a.size()
    device=belong.device
    nn=int(belong.max())+1
    aa=torch.zeros(b,nn,d,device=device)

    idx=belong.unsqueeze(-1).repeat(1,1,d).long()
    aa.scatter_add_(dim=1,index=idx,src=a)

    weight=torch.zeros(b,nn,device=device)
    one=torch.ones(b,n,device=device)
    weight.scatter_add_(dim=1,index=belong.long(),src=one)
    weight=torch.clamp(weight,1)
    aa=aa/weight.unsqueeze(-1)
    return aa,weight

def align(a,m):
    a0=a.detach().clone()
    b,n,d=a0.size()
    device=a0.device
    if(n%m==0):
        return a0
    
    nn=m-(n%m)
    padding=torch.zeros(b,nn,d,device=device)
    a0=torch.cat((a0,padding),1)
    return a0

def place_back(belong,orig_idx):
    b,n=belong.size()
    device=belong.device

    ret=torch.zeros(b,n,device=device,dtype=belong.dtype)

    ret=ret.scatter(1,orig_idx,belong)
    return ret


def discretize(belong):
    b,n=belong.size()
    device=belong.device
    sorted=torch.sort(belong,-1)
    val,idx=sorted.values,sorted.indices
    tmp=((val[:,1:]-val[:,:-1])>0).int()
    
    disc=torch.zeros(b,n,device=device)
    disc[:,1:]=tmp
    disc=torch.cumsum(disc,-1)
    
    for i in range(b):
        belong[i][idx[i]]=disc[i]
    
    return belong

def lsh_sort(a):
    b,n,d=a.size()
    device=a.device
    T=torch.randn(d,device=device)

    proj=(a*T).sum(-1)
    sorted=proj.sort(-1)

    myrange=torch.arange(b,device=device)
    shift=myrange*n

    idx=(sorted.indices+shift.unsqueeze(-1)).view(-1)

    ret=a.view(b*n,d)
    ret=ret[idx].view(b,n,d)

    return ret,sorted.indices

def fliter_head(head_feature,head_siz,head_idx,cnt_thre):
    b=len(head_feature)
    for i in range(b):
        idx=(head_siz[i]>cnt_thre).nonzero().view(-1)
        head_feature[i]=head_feature[i][idx]
        head_siz[i]=head_siz[i][idx]
        head_idx[i]=head_idx[i][idx]
    return head_feature,head_siz,head_idx


def cal_siz_from_belong(belong):
    siz=(belong.unsqueeze(-1)==belong.unsqueeze(-2)).sum(-1)
    return siz

def fetch_head(a,ishead,belong_siz):
    b,_,_,d=a.size()
    a,ishead,belong_siz=a.view(b,-1,d),ishead.view(b,-1),belong_siz.view(b,-1)
    b,n,d=a.size()
    device=a.device

    head_num=ishead.sum(-1)
    N=head_num.max().int()

    head_feature=[]
    head_siz=[]
    head_idx=[]

    for i in range(b):
        idx=ishead[i].nonzero().view(-1)
        head_idx.append(idx)
        head_feature.append(a[i][idx])
        head_siz.append(belong_siz[i][idx])

    return head_feature,head_siz,head_idx