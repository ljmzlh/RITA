import torch.nn as nn
import torch
import math
import time

from torch.nn.modules.activation import Softmax
from myBertUtils import BertSelfOutput
import torch.nn.functional as F
from attentionBlock import va,ga,pa
from performer_pytorch import FastAttention
from utils import glo,speed
from linformer import LinformerSelfAttention



class BertSelfAttention(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.kqv_time=0

        d=self.attention_head_size
        if(glo.flag=='pa'):
            nb=int(d * math.log(d))
            if(args.pa_nb_rate!=-1):
                nb=int(d*args.pa_nb_rate)
            self.fast_attention=FastAttention(dim_heads=d,
                            nb_features=nb,causal = False, 
                            generalized_attention = False)
        elif(glo.flag=='la'):
            self.la_attn=LinformerSelfAttention(dim=config.hidden_size,
                                                seq_len=glo.n+1,heads=self.num_attention_heads,
                                                k=args.la_k,one_kv_head = True,
                                                share_kv = True)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        weight=None
    ):
        n=hidden_states.size(1)
        output_attn=False

        torch.cuda.synchronize()
        st=time.time()

        if(glo.flag!='la'):
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

        torch.cuda.synchronize()
        speed.kqv+=time.time()-st    
        
        
        
    

        if(glo.flag=='va' or (glo.flag=='ga' and glo.N[glo.layer]>n/2)):
            attn,context_layer=va(query_layer,key_layer,value_layer,self.dropout)
        elif(glo.flag=='ga'):
            attn,context_layer=ga(query_layer,key_layer,value_layer,self.dropout,weight)
        elif(glo.flag=='pa'):
            attn,context_layer=pa(query_layer,key_layer,value_layer,self.fast_attention)
        elif(glo.flag=='la'):
            #input hidden; output context_layer
            context_layer=self.la_attn(hidden_states)

        
        
        
        torch.cuda.synchronize()
        speed.attn+=time.time()-st

        if(output_attn):
            glo.attn[glo.layer]=attn[0].cpu().tolist()

        if(glo.flag!='la'):
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

        outputs =  (context_layer,)
        return outputs


class myBertAttention(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.self = BertSelfAttention(config,args)
        self.output = BertSelfOutput(config,args)
        self.pruned_heads = set()
        self.kqv_time=0

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        weight=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            weight
        )
        self.kqv_time=self.self.kqv_time
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs