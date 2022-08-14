import argparse
from main import main
from utils import glo,get_default_para,make_log_name
import torch



def run():
    parser = argparse.ArgumentParser()
    

    ## Default parameters
    parser.add_argument("--cuda_idx",type=int,default=0)

    parser.add_argument('--ksiz', type=int, nargs='+',default=[5])
    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--pad",type=str,default='True')
    parser.add_argument("--prj",type=str,default='TransCNN')

    parser.add_argument("--pre_grp",type=str,default=None)
    parser.add_argument("--pre_grp_sort",action='store_true')
    parser.add_argument("--pre_grp_refine",action='store_true')
    parser.add_argument("--pre_grp_thre",type=float,default=None)
    parser.add_argument("--pre_cnt_rate",type=float,default=0.001)
    parser.add_argument("--pre_grp_seg",type=int,default=None)

    parser.add_argument('--N_policy',type=str,default=None)
    parser.add_argument('--N_list',type=int,nargs='+',default=None)
    parser.add_argument('--N_measurement',type=str,default='L2')
    parser.add_argument('--N_L2_eps',type=float,default=None)
    parser.add_argument('--N_cos_thre',type=float,default=None)
    parser.add_argument('--N_min',type=int,default=10)
    parser.add_argument('--N_kprate_upb',type=float,default=0.9)
    parser.add_argument('--N_kprate_lwb',type=float,default=0.8)
    parser.add_argument('--N_sample',type=int,default=3000)
    parser.add_argument('--N_warmup',type=int,default=0)
    parser.add_argument('--N_init',type=int,default=None)
    parser.add_argument('--N_a',type=float,default=0.5)
    parser.add_argument('--Kmeans_rounds',type=int,default=3)
    # N' = alpha*N + (1-alpha)*N*(1+keep_rate-threshold)

    parser.add_argument("--output_dir",type=str,default='checkpoint')
    parser.add_argument("--fitting_dir",default='fitting',type=str)
    parser.add_argument("--restore_model",type=str,default='None')
    parser.add_argument("--restore_opt",type=str,default='None')
    parser.add_argument("--log_name",type=str,default='None')
    parser.add_argument("--resume_training",action='store_true')
    parser.add_argument("--device",type=str,default='cuda')
    parser.add_argument("--model",type=str,default='BERT')
    parser.add_argument("--few",type=int,default=None)
    parser.add_argument("--data_method",type=str,default='sample')
    parser.add_argument("--pre_overlap",type=float,default=0.8)
    parser.add_argument("--data_overlap",type=float,default=0.5)
    parser.add_argument("--all_seed",type=int,default=10)
    parser.add_argument("--data_seed",type=int,default=10)
    parser.add_argument("--imp_method",type=str,default='rand')
    parser.add_argument("--mask_rate",type=float,default=0.2)

    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--warmup_steps",type=int,default=0)
    parser.add_argument("--log_sample",type=int,default=500)
    parser.add_argument("--dev_sample",type=int,default=50000)
    parser.add_argument("--save_total_limit",type=int,default=3)
    parser.add_argument("--min_dev_per_epoch",type=int,default=1)

    parser.add_argument("--hidden_siz",default=64,type=int)
    parser.add_argument("--intermediate_size",default=256,type=int)
    parser.add_argument("--num_layers",default=8,type=int)
    parser.add_argument("--num_heads",default=2,type=int)

    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    
    
    ## Required parameters
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--mode",type=str,required=True)
    parser.add_argument("--data_size",type=str,required=True)
    parser.add_argument("--pretrained_path",type=str)
    


    ## No required parameters
    parser.add_argument("--data_file",type=str)
    parser.add_argument("--seg_len",type=int)
    parser.add_argument("--cut_len",type=int)
    parser.add_argument("--num_class",type=int)
    parser.add_argument("--num_channel",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--task",type=str)
    parser.add_argument("--epoch",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--pre",type=float)
    parser.add_argument("--train",type=float)
    parser.add_argument("--dev",type=float)


    args = parser.parse_args()

    args.restore_model=args.pretrained_path

    assert args.dataset in ['hhar','rwhar','wisdm','ecg']
    assert args.mode in ['train','dev','pretrain','finetune']
    if(args.data_size==None):
        assert (args.train!=None or args.few!=None) and args.pre!=None and args.dev!=None
    else:
        assert args.data_size in ['full','few']
    #transcnn->pad=true
    assert (args.prj=='TransCNN' or args.pad=='True')

    glo.pre_grp_thre=args.pre_grp_thre
    glo.pre_grp_seg=args.pre_grp_seg
    glo.pre_cnt_rate=args.pre_cnt_rate
    glo.pre_grp_refine=args.pre_grp_refine
    glo.N_min=args.N_min

    

    
    args=get_default_para(args)

    if(args.N_policy=='manual'):
        p=[]
        for i in range(0,len(args.N_list),2):
            p.append([args.N_list[i],args.N_list[i+1]])
        args.N_list=p

    if(args.log_name=='None'):
        args=make_log_name(args)
    
    if(args.device=='cuda'):
        device = torch.device("cuda:"+str(args.cuda_idx))


    if(args.cut_len==None):
        glo.n=args.seg_len//args.stride
    else:
        glo.n=args.cut_len//args.stride
    glo.K_r=args.Kmeans_rounds
    glo.N_policy=args.N_policy
    
    args.log_name='log'
    print('Logging file:',args.log_name)

    print(args.data_size+'-size',args.mode,'on',args.dataset)
    print('Task:',args.task)

    main(args)

if __name__ == "__main__":
    run()