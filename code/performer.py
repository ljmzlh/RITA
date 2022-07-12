import torch
from performer_pytorch import PerformerLM

model = PerformerLM(
    num_tokens = 20000,
    max_seq_len = 2048,             # max sequence length
    dim = 512,                      # dimension
    depth = 12,                     # layers
    heads = 8,                      # heads
    causal = False,                 # auto-regressive or not
    nb_features = 256,              # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
    feature_redraw_interval = 1000, # how frequently to redraw the projection matrix, the more frequent, the slower the training
    generalized_attention = False,  # defaults to softmax approximation, but can be set to True for generalized attention
    kernel_fn = torch.nn.ReLU(),    # the kernel function to be used, if generalized attention is turned on, defaults to Relu
    reversible = True,              # reversible layers, from Reformer paper
    ff_chunks = 10,                 # chunk feedforward layer, from Reformer paper
    use_scalenorm = False,          # use scale norm, from 'Transformers without Tears' paper
    use_rezero = False,             # use rezero, from 'Rezero is all you need' paper
    ff_glu = True,                  # use GLU variant for feedforward
    emb_dropout = 0.1,              # embedding dropout
    ff_dropout = 0.1,               # feedforward dropout
    attn_dropout = 0.1,             # post-attn dropout
    local_attn_heads = 4,           # 4 heads are local attention, 4 others are global performers
    local_window_size = 256,        # window size of local attention
    rotary_position_emb = True,     # use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding
    shift_tokens = True             # shift tokens by 1 along sequence dimension before each block, for better convergence
)