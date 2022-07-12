from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
from myBertUtils import BertSelfOutput
from utils import glo,speed
import json
from grouping import grouping,index_add

def norm(a):
    d=a.size(-1)
    return F.layer_norm(a,[d])/d


class Rel1SelfAttention(nn.Module):
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

        self.grp=True if(args.grp=='True') else False
        self.treat_as_one=True if(args.treat_as_one=='True') else False
        self.qk_mode=args.qk_mode
        self.LN=True if(args.LN=='True') else False

    
    def _rel_shift(self,x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]
        return x
    
        
    def transpose_for_scores(self, x,v=False):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if(v==True):
            return x.permute(0, 2, 1, 3)
        return x.permute(1,0,2,3).contiguous()

    def va(self,hidden_states,r_emb,r_w_bias,r_bias):
            b,n,h,d=hidden_states.size(0),hidden_states.size(1),self.num_attention_heads,self.attention_head_size
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer=self.key(hidden_states) if(self.qk_mode=='qk') else self.query(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            r_emb = r_emb[-n:]
            r_bias = r_bias[-n:]
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer,v=True)
            if(self.LN==True):
                query_layer,key_layer=norm(query_layer),norm(key_layer)
 
            st=time.time()
            rw_head_q = (query_layer + r_w_bias[None]).contiguous()
            AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, key_layer))
            B_ = torch.einsum('ibnd,jnd->ijbn', (query_layer, r_emb))
            D_ = r_bias[None, :, None]
            BD=B_ + D_
            speed.va_matmul+=time.time()-st

            st=time.time()
            BD = self._rel_shift(BD).contiguous()
            attention_scores=torch.add(AC,BD)
            attention_scores=attention_scores.permute(2,3,0,1)
            attention_scores= attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)
            speed.va_add_soft+=time.time()-st

            context_layer = torch.matmul(attention_probs, value_layer)
            
            return attention_probs,context_layer
      
    def ga(self,hidden_states,r_emb,r_w_bias,r_bias):
            b,n,h,d=hidden_states.size(0),hidden_states.size(1),self.num_attention_heads,self.attention_head_size
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer=self.key(hidden_states) if(self.qk_mode=='qk') else self.query(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            r_emb = r_emb[-n:]
            r_bias = r_bias[-n:]

            query_layer = self.transpose_for_scores(mixed_query_layer,v=True)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer,v=True)
            if(self.LN==True):
                query_layer,key_layer=norm(query_layer),norm(key_layer)
            
            gq,belongq,_=grouping(query_layer,'threshold')
           
            st=time.time()
            gq=gq.permute(2,0,1,3)
            rw_head_q = gq + r_w_bias[None]
            AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, key_layer))
            B_ = torch.einsum('ibnd,jnd->ijbn', (gq, r_emb))
            D_ = r_bias[None, :, None]
            BD=B_+D_
            speed.ga_matmul+=time.time()-st
            
            attention_probs=None

            #scatter
            idx=belongq.permute(2,0,1).unsqueeze(1).repeat(1,n,1,1)
            tAC=torch.zeros(n,n,b,h).to(mixed_key_layer.device)
            
            st=time.time()
            AC=AC.contiguous()
            tAC=torch.gather(AC,0,idx)
            speed.AC+=time.time()-st

            tBD=torch.zeros(n,n,b,h).to(mixed_key_layer.device)
            st=time.time()
            BD=BD.contiguous()
            tBD=torch.gather(BD,0,idx)
            speed.BD+=time.time()-st

            st=time.time()
            AC = tAC
            BD = self._rel_shift(tBD)
            attention_scores=torch.add(AC,BD)
            attention_scores=attention_scores.permute(2,3,0,1)
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)
            speed.ga_add_soft+=time.time()-st

            context_layer = torch.matmul(attention_probs, value_layer)
            
            return attention_probs,context_layer



    def gga(self,hidden_states,r_emb,r_w_bias,r_bias):
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer=self.key(hidden_states) if(self.qk_mode=='qk') else self.query(hidden_states)
            mixed_value_layer = self.value(hidden_states) 

            query_layer = self.transpose_for_scores(mixed_query_layer,v=True)
            key_layer=self.transpose_for_scores(mixed_key_layer,v=True)
            value_layer = self.transpose_for_scores(mixed_value_layer,v=True)
            b,h,n,d=query_layer.size()
            if(self.LN==True):
                query_layer,key_layer=norm(query_layer),norm(key_layer)
            device=mixed_query_layer.device

            group,belongq,cntq=grouping(query_layer,'threshold')

            N=cntq[0].size(-1)
            r_emb = r_emb[-N:]
            r_bias = r_bias[-N:]
            gk=index_add(key_layer,belongq,cntq,N)

            gk=gk.permute(2,0,1,3)
            gq=group.permute(2,0,1,3)
            rw_head_q = gq + r_w_bias[None]
            AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, gk))
            B_ = torch.einsum('ibnd,jnd->ijbn', (gq, r_emb))
            D_ = r_bias[None, :, None]
            BD=B_+D_
            
            BD = self._rel_shift(BD)
            attention_scores=AC+BD
            
            attention_scores=attention_scores.permute(2,3,0,1)
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            gv=index_add(value_layer,belongq,cntq,N)
            gcontext_layer = torch.matmul(attention_probs, gv)

            idx=torch.zeros(b,h,n).to(device).long()
            for i in range(b):
                idx[i]=belongq[i].to(device)
            idx=idx.unsqueeze(3).repeat(1,1,1,d)
            context_layer=torch.gather(gcontext_layer,2,idx)
            
            return None,context_layer

    





    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        r_emb=None, r_w_bias=None, r_bias=None
    ):
        output_attn=False
        if(self.treat_as_one==False):
            attention_probs_va,context_layer_va=self.va(hidden_states,r_emb,r_w_bias,r_bias)       
            #attention_probs_ga,context_layer_ga=self.ga(hidden_states,r_emb,r_w_bias,r_bias)
            if(self.grp==True):
                attention_probs,context_layer=attention_probs_ga,context_layer_ga
            else:
                attention_probs,context_layer=attention_probs_va,context_layer_va
                
            if(output_attn):
                glo.attn[glo.layer]=attention_probs[0].cpu().tolist()
        else:
            _,context_layer=self.gga(hidden_states,r_emb,r_w_bias,r_bias)
           
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer,)
    
        return outputs




class Rel1Attention(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.self = Rel1SelfAttention(config,args)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
     

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        r_emb=None, r_w_bias=None, r_bias=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            r_emb, r_w_bias, r_bias
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs