from turtle import forward
from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import TransformerEncoderLayer
import time

from TST_utils import get_pos_encoder,TransformerBatchNormEncoderLayer,_get_activation_fn




def model_factory(args):
    task = args.task
    feat_dim = args.num_channel
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = args.seg_len

    if (task == "cls" or task=='imp'):
        num_labels = args.num_class
        return TSTransformerEncoderClassiregressor(args,feat_dim, max_seq_len, 
                args.hidden_siz, args.num_heads, args.num_layers, 
                args.intermediate_size, num_classes=num_labels,
                )
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))





class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, args, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.cls_output_layer = self.build_output_module(d_model, max_len, num_classes)
        self.imp_output_layer = nn.Linear(d_model, feat_dim)

        self.task=args.task

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        return output_layer


    def forward_run(self, input,args):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        device=input.device
        b,l,d=input.size()

        padding_masks=torch.ones(b,l,device=device)
        padding_masks=padding_masks.bool()
        X=input
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        if(self.task=='cls'):
            output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
            output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
            output = self.cls_output_layer(output)  # (batch_size, num_classes)
        elif(self.task=='imp'):
            output = self.imp_output_layer(output)
        else:
            raise ValueError('sb')

        return output