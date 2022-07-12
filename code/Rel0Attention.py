import torch.nn as nn
import torch
import math
import time
from myBertUtils import BertSelfOutput
from utils import speed

class Rel0SelfAttention(nn.Module):
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
        self.r_net = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.grp=True if(args.grp=='True') else False
        self.treat_as_one=True if(args.treat_as_one=='True') else False
    
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
        if(v):
            return x.permute(0, 2, 1, 3)
        return x.permute(1,0,2,3).contiguous()
        
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        r=None,r_w_bias=None,r_r_bias=None,grp=None
    ):
        n=hidden_states.size(1)
        st=time.time()
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer,v=True)


        st=time.time()
        rlen=r.size(0)
        r_head_k = self.r_net(r)
        rw_head_q = query_layer + r_w_bias[None]
        r_head_k = r_head_k.view(rlen, self.num_attention_heads, self.attention_head_size)
        rw_head_q = query_layer + r_w_bias                      
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, key_layer)).contiguous() 
        rr_head_q = query_layer + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k)) 
        BD = self._rel_shift(BD).contiguous()
        attention_scores=torch.add(AC,BD)
        attention_scores=attention_scores.permute(2,3,0,1)
        speed.va_matmul+=time.time()-st

        st=time.time()
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        speed.va_add_soft +=time.time()-st

        st=time.time()
        context_layer = torch.matmul(attention_probs, value_layer)

        st=time.time()
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs






class Rel0Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = Rel0SelfAttention(config)
        self.output = BertSelfOutput(config)
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
        r=None,
        r_w_bias=None,
        r_r_bias=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            r,r_w_bias,r_r_bias
        )
        self.kqv_time=self.self.kqv_time
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs