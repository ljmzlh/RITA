from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn

from Rel1Layer import Rel1Layer
import time
from utils import glo


class Rel1Encoder(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Rel1Layer(config,args) for _ in range(config.num_hidden_layers)])
        self.attention_time=0

        self.n_layer,self.max_klen=config.num_hidden_layers,config.max_position_embeddings
        self.n_head=config.num_attention_heads
        self.d_head=config.hidden_size//config.num_attention_heads
        self.d_model=config.hidden_size

        self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_layer, self.n_head, self.d_head))
        self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head))

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
        all_hidden_states = ()
        all_attentions = ()
        klen=hidden_states.size(1)




        for i, layer_module in enumerate(self.layer):
            glo.layer=i
            glo
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            r_emb, r_bias = self.r_emb[i], self.r_bias[i]

            layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    r_emb, self.r_w_bias[i],r_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            self.attention_time+=layer_module.attention_time#######
        

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
