from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn

from Rel0Layer import Rel0Layer
import time

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]



class Rel0Encoder(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Rel0Layer(config,args) for _ in range(config.num_hidden_layers)])
        self.attention_time=0
        self.kqv_time=0

        self.n_layer,self.max_klen=config.num_hidden_layers,config.max_position_embeddings
        self.n_head=config.num_attention_heads
        self.d_head=config.hidden_size//config.num_attention_heads
        self.d_model=config.hidden_size
        
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.drop = nn.Dropout(config.hidden_dropout_prob)

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
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        

            pos_seq = torch.arange(klen-1, -1, -1.0, device=hidden_states.device, 
                                   dtype=hidden_states.dtype)
            pos_emb = self.pos_emb(pos_seq)
            pos_emb = self.drop(pos_emb)

            layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    pos_emb,self.r_w_bias,self.r_r_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            self.attention_time+=layer_module.attention_time#######
            self.kqv_time+=layer_module.kqv_time
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
