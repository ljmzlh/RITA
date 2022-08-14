from tkinter import N
import torch
from utils import glo
import time

from pre_grp_utils import all_distance,cal_dis,cal,align,place_back
from pre_grp_utils import discretize,lsh_sort,fliter_head,fetch_head
from pre_grp_utils import cal_siz_from_belong


sb=0


def adjcent(a):
    b,n,d=a.size()
    device=a.device
    thre=glo.pre_grp_thre
    mask=torch.ones(b,n,device=device)
    mask[:,::2]=0

    dis=cal_dis(a[:,1:],a[:,:n-1])
    pad=torch.zeros(b,1,device=device)
    dis=torch.cat([pad,dis],-1)
    merge=(dis<thre)*mask

    belong=torch.ones(b,n,device=device)
    belong[:,0]=0
    belong=belong-merge
    belong=belong.cumsum(-1)

    aa,weight=cal(belong,b,n,d)
    return aa,weight,belong


def cal_dlt_seq(a0,m,thre):
    b,n,d=a0.size()
    device=a0.device
    a=a0.view(b,n//m,m,d).contiguous()
    tmp=a[:,:,0]

    dlt=torch.zeros(b,n//m,m,device=device)
    dlt[:,1:,0]=1

    for i in range(1,m):
        t=a[:,:,i]
        dis=cal_dis(t,tmp)
        split=(dis>thre).int()
        dlt[:,:,i]=split
        split=split.unsqueeze(-1)
        tmp+=split*(t-tmp)

    dlt=dlt.view(b,n).contiguous()
    return dlt


def sequential(a0):
    b,n,d=a0.size()
    device=a0.device
    m=glo.pre_grp_seg if(glo.pre_grp_seg!=None) else 5
    thre=glo.pre_grp_thre if(glo.pre_grp_thre!=None) else 0.1

    dlt=cal_dlt_seq(align(a0,m),m,thre)
    dlt=dlt[:,:n]
    belong=dlt.cumsum(-1)

    aa,weight=cal(belong,a0)
    return aa,weight,belong




def refine(a,belong,belong_siz,head_feature,head_siz,head_idx,dis_thre):
    b,l,m,d=a.size()
    device=a.device
    a,belong_siz=a.view(b,-1,d),belong_siz.view(b,-1)
    b,n,d=a.size()
    cnt_rate=glo.pre_cnt_rate if(glo.pre_cnt_rate!=None) else 0.001
    cnt_thre=int(n*cnt_rate)

    head_feature,head_siz,head_idx=fliter_head(head_feature,head_siz,head_idx,cnt_thre)
    
    proj=torch.arange(l,device=device)*m
    proj=proj.unsqueeze(0).unsqueeze(-1)
    belong=(belong+proj).view(b,n)

    
    for i in range(b):
        minor_idx=(belong_siz[i]<=cnt_thre).nonzero().view(-1)
        if(minor_idx.size(0)==0 or head_feature[i].size(0)==0):
            continue
        minor_feature=a[i][minor_idx]
        minor_belong=belong[i][minor_idx]

        dis=cal_dis(minor_feature,head_feature[i])
        
        mi=dis.min(-1)
        val,idx=mi.values,mi.indices
        mi_head_idx=head_idx[i][idx]

        to_join=(val<dis_thre).int()
        
        new_belong=to_join*mi_head_idx+(1-to_join)*minor_belong
        belong[i][minor_idx]=new_belong
    
    return belong
    

def cal_pairwise_belong(a0,m,thre,pre_sort):
    b,n,d=a0.size()
    device=a0.device
    
    if(pre_sort==True):
        a0,orig_idx=lsh_sort(a0)
    else:
        orig_idx=torch.arange(n,device=device).unsqueeze(0).repeat(b,1)

    a=align(a0,m).view(b,-1,m,d).contiguous()

    #n=l*m
    dis=all_distance(a)
    b,l,m,_=a.size()
    device=a.device

    ishead=torch.zeros(b,l,m,device=device)
    belong=torch.zeros(b,l,m,device=device)
    cnt=torch.zeros(b,l,device=device)

    for i in range(m):
        dt=dis[:,:,i,:]
        dis_to_head=dt*ishead+(1-ishead)*99999

        mi=dis_to_head.min(-1)
        val,idx=mi.values,mi.indices
        
        mergeable=(val<thre).int()
        to_be_head=1-mergeable

        belong[:,:,i]=to_be_head*i+mergeable*idx
        ishead[:,:,i]=to_be_head
        cnt=cnt+to_be_head
    
    belong_siz=cal_siz_from_belong(belong)
    head_feature,head_siz,head_idx=fetch_head(a,ishead,belong_siz)
    
    if(glo.pre_grp_refine==True):
        belong=refine(a,belong,belong_siz,head_feature,head_siz,head_idx,thre)
    else:
        proj=torch.arange(l,device=device)*m
        proj=proj.unsqueeze(0).unsqueeze(-1)
        belong=(belong+proj).view(b,-1)

    belong=discretize(belong)

    belong=belong[:,:n]
    if(pre_sort==True):
        belong=place_back(belong,orig_idx)     
    
    return belong


wl,ms,vs,ls=0,0,0,0


def pairwise(a,pre_sort):
    global sb,wl,ms,vs,ls
    m=glo.pre_grp_seg if(glo.pre_grp_seg!=None) else 5
    thre=glo.pre_grp_thre if(glo.pre_grp_thre!=None) else 0.1
    if(sb==0):
        print('pairwise thre=%f, seg=%d'%(thre,m))
        sb=1

    belong=cal_pairwise_belong(a,m,thre,pre_sort)

    aa,weight=cal(belong,a)
    
    return aa,weight,belong


def pre_grp(a,strategy,pre_sort):
    if(strategy=='seq'):
        return sequential(a)
    elif(strategy=='adj'):
        return adjcent(a)
    elif(strategy=='pai'):
        return pairwise(a,pre_sort)
    else:
        raise ValueError('sb')



if __name__=='__main__':
    b,n,d=3,10,4
    a=torch.rand(b,n,d)
    m=5
    import random
    for j in range(n):
        sb=random.randint(0,3)
        for i in range(b):
            for k in range(d):
                a[i,j,k]=sb
    print(a[0])
    aa,weight,belong=pre_grp(a,'pai',True)
    print(belong)




