from torch.nn.modules import padding
from transformers import BertConfig,BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F



class SIMPLE(nn.Module):
    def __init__(self):
        super(SIMPLE, self).__init__()
        
    
    
    def imputate(self,input,args,mask):
        b,n,d=input.size()
        device=input.device
        output=input.detach().clone()
        for i in range(b):
            last=None
            for j in range(n):
                if(mask[i][j]<0.9):
                    last=output[i][j].detach().clone()

            for j in range(n):
                if(mask[i][j]>0.9):
                    output[i][j]=last.detach().clone()
                else:
                    last=output[i][j].detach().clone()

        return output

    def forward_run(self,input,args,mask):
        if(args.task=='cls'):
            return self.classify(input,args)
        elif(args.task=='for'):
            return self.forcast(input,args)
        elif(args.task=='imp'):
            return self.imputate(input,args,mask)