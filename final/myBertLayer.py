from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import time

from myBertAttention import myBertAttention
from myBertUtils import BertIntermediate,BertOutput

from utils import glo


class myBertLayer(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.attention = myBertAttention(config,args)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = myBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.attention_time=0
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
        
        n=hidden_states.size(1)
        st=time.time()

        
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
            weight=weight)
        
        

        self.kqv_time=self.attention.kqv_time

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
