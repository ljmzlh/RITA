from pretrain import pretrain
from train import train
from dev import dev

from raw_to_input import raw_to_input
from utils import set_all_seed,set_data_seed,glo,fetch_few_data
from dataset import make_dataset,read_dataset,dataset_dist

from myLM import BERT


from tensorboardX import SummaryWriter
from read_data_hhar import read_data_hhar
from read_data_rwhar import read_data_rwhar
from read_data_wisdm import read_data_wisdm
from read_data_ecg import read_data_ecg


def dist(data,args,flag):
    print('Label distribution in',flag,'data:')

    t=[0 for _ in range(args.num_class)]

    batch_size=1024
    st=0
    while(st<data.len):
        ed=min(data.len,st+batch_size)
        n=ed-st
        batch=data.fetch(st,ed,args)
        for i in range(n):
            lab=int(batch['label'][i])
            t[lab]+=1
        st=ed

    cnt=0
    for i in range(args.num_class):
        cnt+=t[i]
    
    print(t,cnt)
    print()

def ddist(dataset_pretrain,dataset_train,dataset_dev,args):
    if(args.task=='for' or args.task=='imp'):
        return
    dist(dataset_pretrain,args,'Pretraining')
    dist(dataset_train,args,'Training')
    dist(dataset_dev,args,'Validation')


def main(args):
    device=args.device


    print('creating model: seed',args.all_seed)
    set_all_seed(args)

    model=BERT(hidden_siz=args.hidden_siz,num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,intermediate_siz=args.intermediate_size,
        num_class=args.num_class,device=device,args=args).to(device)


    print('loading data: seed',args.data_seed)
    set_data_seed(args)
    
    data={'train':None,'dev':None}
    dataset_name='tensor_dataset'
    tensor_path='./'+dataset_name
    import os
    if(os.path.exists(tensor_path)==False):
        print('Can\'t find tensor dataset file')
        print('Reading raw dataset')
        if(args.dataset=='wisdm'):
            data=read_data_wisdm(args)
        elif(args.dataset=='rwhar'):
            data=read_data_rwhar(args)
        elif(args.dataset=='hhar'):
            data=read_data_hhar(args)
        elif(args.dataset=='ecg'):
            data=read_data_ecg(args)
        
        data=raw_to_input(args,data)
        

    glo.N=[glo.n for _ in range(args.num_layers)]
    if(args.N_init==None):
        args.N_init=glo.n
    
    if(args.mode=='finetune'):
        if(args.restore_model=='None'):
            raise ValueError("finetune but not providing pretrained model")

    set_all_seed(args)
    writer= SummaryWriter(args.log_name)

    glo.args=args

    if(os.path.exists(tensor_path)==True):
        raise RuntimeError('Please remove dir ['+tensor_path+']')

    if(os.path.exists(tensor_path)==False):
        os.mkdir('tensor_dataset')
        dataset_pretrain=make_dataset(data['pretrain'],'pretrain',args,dataset_name)
        dataset_train=make_dataset(data['train'],'train',args,dataset_name)
        dataset_dev=make_dataset(data['dev'],'dev',args,dataset_name)
        if(args.task!='cls'):
            dataset_dist(dataset_name)
    else:
        dataset_pretrain=read_dataset(tensor_path,'pretrain',args)
        dataset_train=read_dataset(tensor_path,'train',args)
        dataset_dev=read_dataset(tensor_path,'dev',args)
        if(args.task!='cls'):
            dataset_dist(tensor_path)

    if(args.few!=None):
        dataset_train=fetch_few_data(dataset_train,args)
    ddist(dataset_pretrain,dataset_train,dataset_dev,args)
    
    if(args.mode=='pretrain'):
        pretrain(dataset_pretrain,dataset_dev,args,model,writer,device)
    elif(args.mode=='train' or args.mode=='finetune'):
        train(dataset_train,dataset_dev,args,model,writer,device)
    elif(args.mode=='dev'):
        dev(dataset_dev,args,model,writer,device)