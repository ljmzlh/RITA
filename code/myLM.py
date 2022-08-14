from transformers import BertConfig,BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.parameter import Parameter
from myBertModel import myBertModel
from utils import speed



class Proj(nn.Module):
    def __init__(self,args):
        super(Proj,self).__init__()
        self.trans_cnn=nn.ModuleList([])
        self.args=args
        if(args.prj=='TransCNN'):
            now=0
            c_out=args.hidden_siz
            for sz in reversed(args.ksiz):
                if(now==len(args.ksiz)-1):
                    c_out=args.num_channel
                padding=sz//2 if(args.pad=='True') else 0
                self.trans_cnn.append(nn.ConvTranspose1d(in_channels=args.hidden_siz,
                                                        out_channels=c_out,kernel_size=sz,
                                                        stride=args.stride,padding=padding))
        else:
            self.prj=nn.Linear(args.hidden_siz,3)


    def forward(self,x):
        if(self.args.prj=='TransCNN'):
            x=x.permute(0,2,1)
            for lay in self.trans_cnn:
                x=lay(x)
            x=x.transpose(1,2)
        else:
            x=self.prj(x)
        return x

class TokenEmbedding(nn.Module):
    def __init__(self,args) -> None:
        super(TokenEmbedding,self).__init__()
        self.cnn_layers=nn.ModuleList([])
        c_in=args.num_channel
        for sz in args.ksiz:
            if(args.pad=='True'):
                assert (sz%2)==1
            padding=sz//2 if(args.pad=='True') else 0
            self.cnn_layers.append(nn.Conv1d(in_channels=c_in,out_channels=args.hidden_siz,
                                            kernel_size=sz,stride=args.stride,
                                            padding_mode='circular',padding=padding))
            c_in=args.hidden_siz
    
    def forward(self,x):
        torch.cuda.synchronize()
        st=time.time()

        x=x.permute(0,2,1)
        for lay in self.cnn_layers:
            x=lay(x)
        x=x.transpose(1,2)

        torch.cuda.synchronize()
        speed.cnn+=time.time()-st
        return x

class BERT(nn.Module):
    def init_weight(self,m):
        if(isinstance(m,(nn.Linear,nn.Embedding))):
            m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def __init__(self,hidden_siz,num_hidden_layers,num_attention_heads,intermediate_siz,num_class,device,args):
        super(BERT, self).__init__()
        vocab_siz=1
        config=BertConfig(vocab_size=vocab_siz,hidden_size=hidden_siz,num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,intermediate_size=intermediate_siz,type_vocab_size=4,
        max_position_embeddings=10048,
        )
        self.config=config
        
        self.input_embedding=TokenEmbedding(args)
        self.cls=Parameter(torch.FloatTensor(hidden_siz))
        self.cls.data.normal_(mean=0.0, std=config.initializer_range)

        self.bert = myBertModel(config,args).to(device)
        self.fc= nn.Linear(hidden_siz,num_class).to(device)

        self.prj=Proj(args)

        self.attention_time=0
        self.bert_time=0
        self.bef=0
        self.aft=0
        self.emb=0
        
    
    def get_hidden(self,input,args):
        x=self.input_embedding(input)
        
        n=x.size(0)
        cls=torch.randn(n,1,args.hidden_siz).to(args.device)
        for i in range(n):
            cls[i,0]=self.cls
        #
        x=torch.cat((cls,x),1)
        self.bef=time.time()-self.bef

        st=time.time()
        outputs=self.bert(inputs_embeds=x)
        self.bert_time=time.time()-st
        self.aft=time.time()
        self.attention_time=self.bert.attention_time

        x=outputs[0]
        return x
    
    def classify(self,input,args):
        self.bef=time.time()
        hidden=self.get_hidden(input,args).to(args.device)
        hid=hidden[:,0,:]
        output=self.fc(hid).to(args.device)
        self.aft=time.time()-self.aft
        return output
    
    def forcast(self,input,args):
        hidden=self.get_hidden(input,args).to(args.device)
        hidden=hidden[:,1:,:]
        prj=self.prj(hidden)
        output=prj[:,-7:,:]
        return output
    
    def imputate(self,input,args):
        hidden=self.get_hidden(input,args).to(args.device)
        hidden=hidden[:,1:,:]
        output=self.prj(hidden)
        return output

    def forward_run(self,input,args):
        if(args.task=='cls'):
            return self.classify(input,args)
        elif(args.task=='for'):
            return self.forcast(input,args)
        elif(args.task=='imp'):
            return self.imputate(input,args)

    def pred_mask(self,input,args):
        hidden=self.get_hidden(input,args).to(args.device)
        hidden=hidden[:,1:,:]
        output=self.prj(hidden)
        return output
    
