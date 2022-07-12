import random
import time
import json
import sys
import pynvml

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


from utils import glo, my_save, output_speed, speed, tmp, my_restore
from utils import update_warming_N, my_linear_schedule,N_fiter
from utils import upd_rec,write_rec,fitting,cal_loss,upd_right,process_bar
from utils import get_cuda,clean_cuda

grads = {}

def save_grad(name):
    print('fuck')
    def hook(grad):
        grads[name] = grad
    return hook


acc_loss,global_sample,last_log,last_dev,last_N=0,0,0,0,0
training_time,dev_time,N_time=0,0,0
pre_grp_n=0
dev_min=1e9
dev_acc=0


glo_epoch=0

def adjust_batch_size(args):
    global glo_epoch
    if(args.N_flag=='ga' and glo.warming==False):
        b=None
        if(args.N_policy=='auto'):
            n=0
            for i in range(args.num_layers):
                n+=glo.N[i]
            b=fitting(n)
            print(n,b)
        else:
            b=args.batch_size
        glo.batch_size=max(int(b),1)
    else:
        glo.batch_size=args.batch_size



 
def dev_epoch(args,dataset_dev,model,writer):
    glo.measure_kmeans=False
    global dev_min,dev_acc

    model.eval()
    rec={}
    adjust_batch_size(args)
    with torch.no_grad():
        print('dev begin')
        dev_loss=0
        batch_st=0
        while(batch_st<dataset_dev.len):
            batch_ed=min(dataset_dev.len,batch_st+glo.batch_size)
            #batch=collate(data_dev[batch_st:batch_ed],args)
            batch=dataset_dev.fetch(batch_st,batch_ed,args)

            input,label=batch['input'].to(args.device),batch['label'].to(args.device)
            mask=batch['mask'].to(args.device)

            pred=model.forward_run(input=input,args=args)

            loss=cal_loss(pred,label,args.device,args,mask)

            dev_loss+=loss*(batch_ed-batch_st)

            dlt=upd_right(pred,label,mask,args)
            rec=upd_rec(rec,dlt)

            batch_st=batch_ed

        dev_loss/=dataset_dev.len

    if(dev_loss<dev_min):
        f=open('res_dev','w')
        import json
        f.write(json.dumps(glo.res))
        f.close()

        dev_min=dev_loss
        my_save(ckp_prefix='dev_best',model=model,args=args,optimizer=None,
                scheduler=None,epoch_done=None,for_resume=False,rotate=False,
                global_sample=None,last_log=None,last_dev=None,last_N=None,
                training_time=None)

    print(dev_loss,dev_min)
    writer.add_scalar('dev_loss',dev_loss,global_sample)
    write_rec(rec,writer,global_sample,'dev')

    if(rec.get('acc')!=None):
        acc=1.0*rec['acc'][0]/rec['acc'][1]
        if(dev_acc<acc):
            dev_acc=acc
        print(acc,dev_acc)

    glo.measure_kmeans=True

p_data_time,train_epoch_time,bsz_it=0,0,0


def train_epoch(now_epoch,dataset_train,dataset_dev,args,
                model,optimizer,scheduler,writer,device):
    global global_sample,acc_loss,grads,last_N,last_log,last_dev,pre_grp_n
    global training_time,dev_time,N_time,p_data_time,train_epoch_time
    global bsz_it

    if(glo.flag=='ga'):
        last_N=update_warming_N(args,global_sample,last_N,now_epoch,writer)

    model.train()
    optimizer.zero_grad()
    tot_loss=0

    print('train ',now_epoch)

    torch.autograd.set_detect_anomaly(True)

    batch_st=0
    cuda_mx=0
    rec={}

    tolerance=False
    if(args.N_flag=='ga' and args.N_policy=='auto'):
        tolerance=True
    
    while(batch_st<dataset_train.len):
        training_time_st=None

        if(tolerance):
            try:
                p_data_st=time.time()
                adjust_batch_size(args)
                batch_ed=min(dataset_train.len,batch_st+glo.batch_size)
                batch=dataset_train.fetch(batch_st,batch_ed,args)
                p_data_time+=time.time()-p_data_st

                torch.cuda.synchronize()
                training_time_st=time.time()
                st=time.time()
                input,label=batch['input'].to(device),batch['label'].to(device)
                mask=batch['mask'].to(device)
                torch.cuda.synchronize()
                speed.togpu+=time.time()-st

                torch.cuda.synchronize()
                stt=time.time()

                pred=model.forward_run(input=input,args=args)
            except:
                print(glo.batch_size,'fail')
                glo.fiter.coefficient=glo.fiter.coefficient*0.9
                model.zero_grad()
                continue
        else:
            p_data_st=time.time()
            adjust_batch_size(args)
            batch_ed=min(dataset_train.len,batch_st+glo.batch_size)
            batch=dataset_train.fetch(batch_st,batch_ed,args)
            p_data_time+=time.time()-p_data_st

            torch.cuda.synchronize()
            training_time_st=time.time()
            st=time.time()
            input,label=batch['input'].to(device),batch['label'].to(device)
            mask=batch['mask'].to(device)
            torch.cuda.synchronize()
            speed.togpu+=time.time()-st

            torch.cuda.synchronize()
            stt=time.time()

            pred=model.forward_run(input=input,args=args)
            print(batch_st,torch.any(torch.isnan(pred)))

        
        glo.fiter.coefficient=max(glo.fiter.coefficient,1)


        if(args.pre_grp!=None):
            pre_grp_n+=glo.pre_grp_n*(batch_ed-batch_st)
        bsz_it+=1
        torch.cuda.synchronize()
        speed.fw+=time.time()-stt

        bo=True
        if(torch.any(torch.isnan(pred))):
            '''my_save(ckp_prefix='nan',model=model,args=args,optimizer=optimizer,
                scheduler=scheduler,epoch_done=now_epoch,for_resume=True,rotate=True,
                global_sample=global_sample,last_log=last_log,last_dev=last_dev,
                last_N=last_N,training_time=training_time)
            with open('nan_input','w') as f:
                output=json.dumps(input.detach().cpu().tolist())
                f.write(output)'''
            print('fuck')
            bo=False

        if(bo):
            stt=time.time()
            loss=cal_loss(pred,label,device,args,mask)
                
            torch.cuda.synchronize()
            speed.loss+=time.time()-stt

            assert args.gradient_accumulation_steps==1
            
            if(args.ins_len==None):
                tot_loss+=loss
                acc_loss+=loss.item()*(batch_ed-batch_st)
            
            torch.cuda.synchronize()
            stt=time.time()
            training_time+=time.time()-training_time_st

            cuda_mx=max(cuda_mx,get_cuda(args))

            training_time_st=time.time()
            loss.backward()
            torch.cuda.synchronize()
            speed.bp+=time.time()-stt

            if(args.ins_len==None):
                dlt=upd_right(pred,label,mask,args)
                rec=upd_rec(rec,dlt)

            torch.cuda.synchronize()
            stt=time.time()
            if(glo.measure_pregrp==False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
        model.zero_grad()
        global_sample+=batch_ed-batch_st
        torch.cuda.synchronize()
        speed.upd+=time.time()-stt
        training_time+=time.time()-training_time_st

        if(args.ins_len==None):
            if((global_sample-last_log)>args.log_sample):
                bsz_avg=1.0*(global_sample-last_log)/bsz_it
                bsz_it=0
                if(args.pre_grp!=None):
                    pre_grp_n/=(global_sample-last_log)

                    sba=glo.n/pre_grp_n*0.9
                    sbb=0.9/max(0.75,cuda_mx)
                    ca=sba
                    cb=glo.fiter.coefficient*sbb

                    glo.fiter.coefficient=max(1,min(ca,cb))
                    writer.add_scalar('pre_grp_n',pre_grp_n,global_sample)
                    writer.add_scalar('pre_grp_time',speed.pre_grp,global_sample)
                    pre_grp_n=0
                
                if(args.device=='cuda' and args.N_flag=='ga' and args.N_policy=='auto'):
                    writer.add_scalar('gpu',cuda_mx,global_sample)
                    clean_cuda(args)
                    cuda_mx=0
                    
                acc_loss/=(global_sample-last_log)
                writer.add_scalar('train_loss',acc_loss,global_sample)
                writer.add_scalar('training_time',training_time,global_sample)
                writer.add_scalar('dev_time',dev_time,global_sample)
                writer.add_scalar('N_time',N_time,global_sample)
                writer.add_scalar('p_data_time',p_data_time,global_sample)
                writer.add_scalar('train_epoch_time',train_epoch_time,now_epoch)
                writer.add_scalar('bsz',bsz_avg,global_sample)

                writer.add_scalar('scat_bef_cnt',glo.scat_bef_cnt,global_sample)
                
                glo.tot_N,glo.tot_n=0,0
                acc_loss=0
                last_log=global_sample

                

            dev_time_st=time.time()

            if((global_sample-last_dev)>args.dev_sample):
                dev_epoch(args,dataset_dev,model,writer)
                model.train()
                my_save(ckp_prefix='ckp_step',model=model,args=args,optimizer=None,
                    scheduler=None,epoch_done=None,for_resume=False,rotate=True,
                    global_sample=None,last_log=None,last_dev=None,last_N=None,
                    training_time=None)
                last_dev=global_sample

            dev_time+=time.time()-dev_time_st
            
            N_time_st=time.time()
            if(glo.flag=='ga'):
                last_N=update_warming_N(args,global_sample,last_N,now_epoch,writer)
            N_time+=time.time()-N_time_st

        batch_st=batch_ed
        process_bar(1.0*batch_ed/dataset_train.len)
        torch.cuda.synchronize()
        speed.all+=time.time()-st

    my_save(ckp_prefix='ckp_epoch',model=model,args=args,optimizer=optimizer,
            scheduler=scheduler,epoch_done=now_epoch,for_resume=True,rotate=True,
            global_sample=global_sample,last_log=last_log,last_dev=last_dev,
            last_N=last_N,training_time=training_time)
    
    if(args.ins_len==None):
        write_rec(rec,writer,global_sample,'train')
    else:
        output_speed()




def train(dataset_train,dataset_dev,args,model,writer,device):
    pynvml.nvmlInit()
    glo.measure_kmeans=True

    global global_sample,last_log,last_dev,last_N

    if(args.ins_len!=None):
        args.epoch=1

    optimizer_grouped_parameters = get_paras(model,args)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    t_total = dataset_train.len // 10 // args.gradient_accumulation_steps * args.epoch

    #scheduler = get_linear_schedule_with_warmup(
    #    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    scheduler = my_linear_schedule(optimizer,t_total)
    global training_time
    model,optimizer,scheduler,st_epoch,st_global,st_log,st_dev,st_N,training_time=my_restore(
                                                args,model,optimizer,scheduler)
    global_sample,last_log,last_dev,last_N=st_global,st_log,st_dev,st_N

    max_dev_sample=dataset_train.len // args.min_dev_per_epoch
    args.dev_sample=min(args.dev_sample,max_dev_sample)
    args.log_sample=min(args.log_sample,max_dev_sample)

    print('dev sample:',args.dev_sample)

    glo.fiter=N_fiter(args)


    debug=False
    if(debug==True):
        args.restore_model='checkpoint/nan'
        model,optimizer,scheduler,st_epoch,st_global,st_log,st_dev,st_N,training_time=my_restore(
                                                args,model,optimizer,scheduler)
        run_debug(model,args)
        return


    global train_epoch_time,p_data_time,glo_epoch
    for i in range(st_epoch,args.epoch):
        glo_epoch=i
        st=time.time()
        if(args.task=='imp'):
            train_epoch(i,dataset_train,dataset_dev,args,model,optimizer,scheduler,writer,device)
        else:
            train_epoch(i,dataset_train,dataset_dev,args,model,optimizer,scheduler,writer,device)
        train_epoch_time+=time.time()-st



def run_debug(model,args):
    f=open('nan_input','r')
    input=torch.tensor(json.loads(f.read())).to('cuda')

    pred=model.forward_run(input=input,args=args)
    print(torch.any(torch.isnan(pred)))















def shuffle_list(a):
    n=len(a)
    for i in range(1,n):
        j=random.randint(0,i-1)
        a[i],a[j]=a[j],a[i]
    return a



def get_paras(model,args):
    no_decay = ["bias", "LayerNorm.weight"]
    if(args.only_head==True):
        for n,p in model.fc.named_parameters():
            print(n)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.fc.named_parameters() if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0,
            }
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    return optimizer_grouped_parameters