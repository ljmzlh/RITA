import argparse
import os
from typing import Type
import numpy as np
import torch
import random
from main import main
from utils import glo

'''cpu_num = 2 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)'''



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_idx",type=int,default=None)
    parser.add_argument("--ins_len",type=int,default=None)
    parser.add_argument("--pos_mode",type=str,default=None)
    parser.add_argument('--grp',type=str,default='False')

    parser.add_argument("--random_per_epoch",type=str,default='False')
    parser.add_argument('--ksiz', type=int, nargs='+',default=None)

    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--pad",type=str,default='True')
    parser.add_argument("--prj",type=str,default='TransCNN')

    parser.add_argument("--pre_grp",type=str,default=None)
    parser.add_argument("--pre_grp_sort",action='store_true')
    parser.add_argument("--pre_grp_refine",action='store_true')
    parser.add_argument("--pre_grp_thre",type=float,default=None)
    parser.add_argument("--pre_cnt_rate",type=float,default=0.001)
    parser.add_argument("--pre_grp_seg",type=int,default=None)

    ## N parameters
    parser.add_argument('--N_flag',type=str)
    parser.add_argument('--N_policy',type=str)
    parser.add_argument('--N_measurement',type=str)
    parser.add_argument('--N_list',type=int,nargs='+',default=None)
    parser.add_argument('--N_cos_thre',type=float,default=None)
    parser.add_argument('--N_L2_eps',type=float,default=None)
    parser.add_argument('--N_min',type=int,default=None)
    parser.add_argument('--N_kprate_upb',type=float,default=None)
    parser.add_argument('--N_kprate_lwb',type=float,default=None)
    parser.add_argument('--N_sample',type=int)
    parser.add_argument('--N_warmup',type=int,default=0)
    parser.add_argument('--N_init',type=int,default=None)
    parser.add_argument('--N_a',type=float,default=0)
    parser.add_argument('--K_r',type=int,default=11)
    # N' = alpha*N + (1-alpha)*N*(1+keep_rate-threshold)

    ## baseline para
    parser.add_argument('--pa_nb_rate',type=float,default=None)
    parser.add_argument('--la_k',type=int,default=None)

    ## Required parameters
    parser.add_argument("--dataset",type=str,default=None)
    parser.add_argument("--seg_len",type=int,default=None)
    parser.add_argument("--cut_len",type=int,default=None)
    parser.add_argument("--output_dir",type=str,default='checkpoint')
    parser.add_argument("--epoch",type=int,default=50)
    parser.add_argument("--restore_model",type=str,default='None')
    parser.add_argument("--restore_opt",type=str,default='None')
    parser.add_argument("--log_name",type=str,default='None')
    parser.add_argument("--mode",type=str,default='train')
    
    parser.add_argument("--resume_training",action='store_true')
    parser.add_argument("--device",type=str,default='cuda')
    
    parser.add_argument("--model",type=str,default='MLP')
    
    parser.add_argument("--pre",type=float)
    parser.add_argument("--train",type=float)
    parser.add_argument("--dev",type=float)
    parser.add_argument("--few",type=int,default=None)

    parser.add_argument("--only_head",action='store_true')

    parser.add_argument("--data_method",type=str)

    parser.add_argument("--pre_overlap",type=float)
    parser.add_argument("--data_overlap",type=float)

    parser.add_argument("--all_seed",type=int,default=10)
    parser.add_argument("--data_seed",type=int,default=10)
    
    ## Model hyperparameters
    parser.add_argument("--num_class",default=None,type=int)
    parser.add_argument("--num_channel",default=None,type=int)

    parser.add_argument("--batch_size",default=8,type=int)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--warmup_steps",type=int,default=0)
    parser.add_argument("--log_sample",type=int,default=500)
    parser.add_argument("--dev_sample",type=int,default=50000)
    parser.add_argument("--save_total_limit",type=int,default=3)
    parser.add_argument("--min_dev_per_epoch",type=int,default=1)

    parser.add_argument("--hidden_siz",default=64,type=int)
    parser.add_argument("--num_layers",default=1,type=int)
    parser.add_argument("--num_heads",default=1,type=int)
    parser.add_argument("--intermediate_size",default=256,type=int)

    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--lr",default=0.00001,type=float)

    parser.add_argument("--task",default='cls',type=str)
    parser.add_argument("--cls_method",default='cls',type=str)
    parser.add_argument("--imp_method",default=None,type=str)
    parser.add_argument("--mask_rate",default=None,type=float)

    parser.add_argument("--norm",default=None,type=str)

    args = parser.parse_args()

    assert args.cls_method in ['cls','mean','max']

    assert ((args.N_flag!='ga') or 
            (args.N_policy=='manual' and args.N_list!=None) or 
            (args.N_policy=='auto' and (args.N_cos_thre!=None or args.N_L2_eps!=None) and args.N_kprate_lwb!=None and args.N_sample!=None))
        
    assert ((args.N_flag!='pa') or (args.pa_nb_rate!=None))

    glo.flag=args.N_flag
    glo.pre_grp_thre=args.pre_grp_thre
    glo.pre_grp_seg=args.pre_grp_seg
    glo.pre_cnt_rate=args.pre_cnt_rate
    glo.pre_grp_refine=args.pre_grp_refine
    glo.N_min=args.N_min

    if(args.dataset=='rwhar'):
        args.datafile='rwhar_scale'
        args.num_class=8
        args.num_channel=3
    elif(args.dataset=='hhar'):
        args.datafile='hhar'
        args.num_class=5
        args.num_channel=3
    elif(args.dataset=='wisdm'):
        args.datafile='accel_watch'
        args.num_class=18
        args.num_channel=3
    elif(args.dataset=='ecg'):
        args.datafile='ecg'
        args.num_class=9
        args.num_channel=12
    elif(args.dataset=='stock'):
        args.num_class=10
        args.num_channel=1
    elif(args.dataset=='mgh'):
        args.num_class=10
        args.num_channel=21

    if(args.cut_len==None):
        glo.n=args.seg_len//args.stride
    else:
        glo.n=args.cut_len//args.stride
    
    if(args.N_policy=='manual'):
        p=[]
        for i in range(0,len(args.N_list),2):
            p.append([args.N_list[i],args.N_list[i+1]])
        args.N_list=p

    if(args.log_name=='None'):
        tr=str(args.train) if(args.few==None) else str(args.few)
        s=str(args.pre)+'-'+tr+'-'+str(args.dev)
        t=str(args.pre_overlap)+'-'+str(args.data_overlap)+'-'+args.data_method
        v='seed'+str(args.all_seed)
        w=('ksiz'+str(args.ksiz)+'_stride'+str(args.stride)+'_pad-'+str(args.pad)+
            '-'+args.prj)
        task=args.task+(
            ('ON'+args.cls_method if(args.task=='cls') else '')+
            (args.imp_method+str(args.mask_rate) if(args.task=='imp') else '')
            )
        norm=str(args.norm)+'NORM'
        dataset=args.dataset+(str(args.seg_len) if(args.cut_len==None) else str(args.cut_len))
        args.log_name=args.mode+'-'+task+'_'+norm+'_'+dataset+'_'+t+'_'+s+'_'+w+'_'+v+'_'+str(args.pos_mode)
        
        model_name='_'+args.model

        if(args.model=='BERT'):
            if(args.N_flag=='va'):
                model_name='_VA'
            elif(args.N_flag=='ga'):
                model_name='_GA-'+args.N_policy+'-'
                if(args.N_policy=='manual'):
                    model_name+=str(args.N_list)
                else:
                    model_name+=(args.N_measurement+'-'+
                                    (str(args.N_cos_thre) if(args.N_measurement=='cos') else str(args.N_L2_eps))+
                                    '-min'+str(args.N_min)+
                                    '-kprate'+str(args.N_kprate_upb)+str(args.N_kprate_lwb)+
                                    '-alpha'+str(args.N_a)+
                                    '-Kround'+str(args.K_r))
            elif(args.N_flag=='pa'):
                model_name='_PA-'+str(args.pa_nb_rate)
            elif(args.N_flag=='la'):
                model_name='_LA-'+str(args.la_k)
        
        model_name=model_name.replace(" ","")

        args.model_name=model_name
        args.log_name+=model_name
        
        if(args.pre_grp!=None):
            args.log_name+=('_pregrp-'+args.pre_grp+'-'+str(args.pre_grp_thre)
                            +str(args.pre_grp_seg))
            if(args.pre_grp_sort==True):
                args.log_name+='-sort'
            if(args.pre_grp_refine==True):
                args.log_name+='-refine'+str(args.pre_cnt_rate)
        if(args.only_head==True):
            args.log_name+='_onlyhead'
        if(args.resume_training==True):
            args.log_name+='_resume'
        
        

    
    glo.K_r=args.K_r
    glo.N_flag,glo.N_policy=args.N_flag,args.N_policy
        
    #transcnn->pad=true
    assert (args.prj=='TransCNN' or args.pad=='True')
    
    args.log_name=args.log_name.replace(' ','')
    print(args.log_name)

    main(args)

if __name__ == "__main__":
    run()