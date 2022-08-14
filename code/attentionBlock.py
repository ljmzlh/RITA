import torch
import torch.nn as nn
import time
from utils import index_add, speed,glo,average,group_measure
from grouping import grouping
import math
from performer_pytorch import FastAttention,PerformerEncDec



def va(query_layer,key_layer,value_layer,dropout):
        b,h,n,d=key_layer.size()

        torch.cuda.synchronize()
        st=time.time()
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(d)
        torch.cuda.synchronize()
        speed.matmul+=time.time()-st

        torch.cuda.synchronize()
        st=time.time()
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attn = dropout(attention_probs)
        torch.cuda.synchronize()
        speed.softmax+=time.time()-st

        torch.cuda.synchronize()
        st=time.time()
        context_layer = torch.matmul(attn, value_layer)
        torch.cuda.synchronize()
        speed.cal_v+=time.time()-st
        return attention_probs,context_layer
    


def ga(query_layer,key_layer,value_layer,dropout,weight):
        b,h,n,d=key_layer.size()
        N=int(glo.N[glo.layer])
        
        torch.cuda.synchronize()
        st=time.time()
        belong,cnt=None,None
        belong,cnt,mask,compatness=grouping(key_layer,'kmeans',N,weight)
        glo.tot_N+=cnt.size(-1)
        torch.cuda.synchronize()
        speed.grp+=time.time()-st
        
        torch.cuda.synchronize()
        st=time.time()
        key_layer=average(key_layer,belong,cnt,mask,average=True,weight=weight)

        if(glo.measure_kmeans==True):
            group_measure(key_layer,compatness,query_layer)

        value_layer=average(value_layer,belong,cnt,mask,average=False,weight=weight)

        torch.cuda.synchronize()
        speed.avg_v+=time.time()-st
        
        torch.cuda.synchronize()
        st=time.time()
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(d)
        torch.cuda.synchronize()
        speed.matmul+=time.time()-st

        scatter_before=False
        if(attention_scores.max()>90):
            attention_scores=attention_scores.double()
            glo.scat_bef_cnt+=1
        if(scatter_before==True):
            idx=belong.unsqueeze(2).repeat(1,1,n,1).to('cuda')
            attention_scores =torch.gather(attention_scores,3,idx)
            
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attn = dropout(attention_probs)

            context_layer = torch.matmul(attn, value_layer)
        else:
            torch.cuda.synchronize()
            st=time.time()
            attn=attention_scores.exp()
            fm=attn*cnt.unsqueeze(2)
            fm=torch.sum(fm,3).unsqueeze(2)
            attention_probs=(attn.permute(0,1,3,2)/fm).permute(0,1,3,2)
            torch.cuda.synchronize()
            speed.softmax+=time.time()-st

            torch.cuda.synchronize()
            st=time.time()
            context_layer=torch.matmul(attention_probs.float(),value_layer)
            torch.cuda.synchronize()
            speed.cal_v+=time.time()-st

        return attention_probs,context_layer




def pa(query_layer,key_layer,value_layer,fast_attention):
    b,h,n,d=key_layer.size()
    context_layer = fast_attention(query_layer, key_layer, value_layer)
    return None,context_layer

