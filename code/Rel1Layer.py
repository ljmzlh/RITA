from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import time

from torch.serialization import validate_cuda_device

from myBertUtils import BertIntermediate,BertOutput
from Rel1Attention import Rel1Attention


class Rel1Layer(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.attention = Rel1Attention(config,args)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.attention_time=0

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
        n=hidden_states.size(1)
        st=time.time()
        
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
            r_emb=r_emb,r_w_bias=r_w_bias,r_bias=r_bias)

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
