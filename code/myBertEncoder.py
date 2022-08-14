import torch
import torch.nn as nn

from myBertLayer import myBertLayer
import time
from utils import glo, speed
from pre_grp import pre_grp



class myBertEncoder(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([myBertLayer(config,args) for _ in range(config.num_hidden_layers)])
        self.attention_time=0
        self.kqv_time=0
        self.pre_grp=args.pre_grp
        self.pre_grp_sort=args.pre_grp_sort

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False
    ):
        self.attention_time=0
        self.kqv_time=0
        all_hidden_states = ()
        all_attentions = ()
        klen=hidden_states.size(1)
        weight=None
        if(self.pre_grp!=None):
            torch.cuda.synchronize()
            st=time.time()
            hidden_states,weight,belong=pre_grp(hidden_states,self.pre_grp,self.pre_grp_sort)
            glo.pre_grp_n=hidden_states.size(1)
            torch.cuda.synchronize()
            speed.pre_grp+=time.time()-st

        for i, layer_module in enumerate(self.layer):
            glo.layer=i
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    weight)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            self.attention_time+=layer_module.attention_time#######
            self.kqv_time+=layer_module.kqv_time
        

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if(self.pre_grp!=None):
            hidden_states=recover(hidden_states,belong)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def recover(a,belong):
    b,n,d=a.size()
    device=a.device

    ret=torch.zeros(b,n,d,device=device)

    idx=belong.unsqueeze(-1).repeat(1,1,d).long()
    ret=torch.gather(input=a,dim=1,index=idx)

    return ret