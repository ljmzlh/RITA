import os
import numpy as np
import torch
import random
import glob
import re
import shutil
import time
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum,row_norms
import numbers
import torch.nn.functional as F
import math

from torch.optim.lr_scheduler import LambdaLR
from dataset import make_dataset_from_tensor

class glo():
    res=None
    pre_grp_refine=None
    pre_cnt_rate=None
    measure_kmeans=True
    f,layer=None,None
    attn,id=None,None
    tot_n,tot_N=0,0
    N,n=None,None
    batch_size=0
    warming=True
    K_r=None
    N_policy=None
    N_min=None
    pre_grp_thre=None
    pre_grp_n=None
    pre_grp_seg=None

    use_mask=True
    total_N,keep_N=[],[]
    keep_rate_list=[]
    split_list,merge_list=[],[]
    args=None

    fiter=None

    scat_bef_cnt=0

class speed():
    all,glo=0,0
    fa=0
    cnn=0
    fw,loss,attn,togpu,bp,upd=0,0,0,0,0,0
    hardcode=0
    inout,inout_st=0,0
    kqv=0
    grp,matmul,avg_k,avg_v,softmax,cal_v=0,0,0,0,0,0
    grp_zeros,grp_ini,grp_dis,grp_min,grp_cnt,grp_upd=0,0,0,0,0,0
    grp_sklearn=0
    get_cnt_zeros,get_cnt_scatter=0,0
    idx_add_zeros,idx_add_idxadd,idx_add_div=0,0,0
    grp_minus,grp_square,grp_sum=0,0,0
    grp_matmul=0
    init_dis,init_upd,init_fetch,init_max=0,0,0,0
    init_a2,init_b2,init_fz,init_zeros=0,0,0,0
    init_sb=0
    pre_grp=0


from N_fit import func_3,func_dui,func_zhi,get_val
import json
class N_fiter:
    def __init__(self,args):
        if(args.N_policy=='auto'):
            mapping={'3':(func_3,'3'),'dui':(func_dui,'dui'),'zhi':(func_zhi,'zhi')}
            path=args.fitting_dir+'/fitting_'+str(int(glo.n))
            t=json.loads(open(path,'r').readline())
            flist=t['flist']

            self.flist=[]
            for f in flist:
                func=mapping[f['func']]
                para=f['para']
                self.flist.append((func,para))
            
            self.rlist=t['rlist']

        self.coefficient=1


    def fit(self,N,scale=False):
            ret=None
            N=max(N,self.rlist[-1][0])
            ret=get_val(self.flist[-1][0],[N],self.flist[-1][1])[0]
            for i in range(len(self.rlist)):
                if(N>self.rlist[i][0]):
                    ret=get_val(self.flist[i][0],[N],self.flist[i][1])[0]
                    break
            if(scale==True):
                ret*=self.coefficient
            return ret




def cal_len2(x):
    a2=torch.square(x).sum(-1)
    return a2

def group_distance(x,measurement):
    dist,a2=None,None
    if(measurement=='L2'):
        ab=2*torch.matmul(x,x.transpose(-1,-2))
        a2=torch.square(x).sum(-1)
        b2=torch.square(x).sum(-1)
        dist=a2.unsqueeze(3)+b2.unsqueeze(2)-ab
    elif(measurement=='dot'): 
        dist=((x[:,:,:, None, :] * x[:,:,None, :, :]).sum(-1))
    elif(measurement=='cos'):
        dist=torch.matmul(x,x.transpose(-1,-2))
        x2,c2=torch.square(x).sum(-1),torch.square(x).sum(-1)
        x2,c2=torch.sqrt(x2),torch.sqrt(c2)
        dist=dist/x2.unsqueeze(3)/c2.unsqueeze(2)
    
    return dist,a2



def kmeans_distance(x,c,measurement='L2',return_compatness=False):
    torch.cuda.synchronize()
    b,h,n,d=x.size()
    _,_,N,_=c.size()
    device=x.device
    sbt=time.time()
    compatness_sum=None

    if(measurement=='L2'):
        torch.cuda.synchronize()
        st=time.time()
        ab=2*torch.matmul(x,c.transpose(-1,-2))
        torch.cuda.synchronize()
        speed.grp_matmul+=time.time()-st

        torch.cuda.synchronize()
        st=time.time()
        a2=torch.square(x).sum(-1)
        b2=torch.square(c).sum(-1)
        torch.cuda.synchronize()
        speed.grp_square+=time.time()-st

        torch.cuda.synchronize()
        st=time.time()
        dist=a2.unsqueeze(3)+b2.unsqueeze(2)-ab
        torch.cuda.synchronize()
        speed.grp_sum+=time.time()-st
    elif(measurement=='dot'):
        torch.cuda.synchronize()
        st=time.time()
        dist=((x[:,:,:, None, :] * c[:,:,None, :, :]).sum(-1))
        torch.cuda.synchronize()
        speed.grp_dis+=time.time()-st
    elif(measurement=='cos'):
        dist=torch.matmul(x,c.transpose(-1,-2))
        x2,c2=torch.square(x).sum(-1),torch.square(c).sum(-1)
        x2,c2=torch.sqrt(x2),torch.sqrt(c2)
        dist=dist/x2.unsqueeze(3)/c2.unsqueeze(2)
    
    torch.cuda.synchronize()
    speed.grp_dis+=time.time()-sbt

    torch.cuda.synchronize()
    st=time.time()
    mi=dist.min(3)
    belong=mi.indices
    belong=dist.argmin(3)
    if(glo.use_mask==True):
        belong=belong.int()
    else:
        belong=belong.long()

    if(return_compatness==True):
        d=mi.values
        compatness_sum=torch.zeros(b,h,N,device=device)
        compatness_sum.scatter_add_(dim=2,index=belong.long(),src=d)

    torch.cuda.synchronize()
    speed.grp_min+=time.time()-st

    return belong,compatness_sum



def average(a,belong,cnt,mask,average,weight):
    N=cnt.size(-1)
    ret=None
    
    ret=torch.einsum ('bhij, bhik -> bhjk', mask, a)
    
    if(average==True):
        torch.cuda.synchronize()
        st=time.time()
        ct=torch.clamp(cnt,1).unsqueeze(3)
        ret=ret/ct
        torch.cuda.synchronize()
        speed.idx_add_div+=time.time()-st

    return ret

def my_max(a,b):
    return torch.abs(a+b)/2+torch.abs(a-b)/2 


def get_cnt(belong,mask,N,weight):
    b,h,n=belong.size()
    device=belong.device
    
    if(weight!=None):
        mask=mask*weight.unsqueeze(1).unsqueeze(-1)
    cnt=mask.sum(2)
    
    cnt=my_max(cnt,torch.ones(b,h,N,device=device))

    return cnt


def discretize(a):
    a=a.int()
    b,h,n=a.size()
    device=a.device
    v,idx=torch.sort(a,dim=-1)
    ret=torch.zeros(b,h,n,device=device).int()
    for i in range(b):
        for j in range(h):
            t=v[i,j,1:]-v[i,j,:n-1]
            dlt=(t>0).int()
            s=torch.cumsum(dlt,dim=-1).int()
            ret[i,j,idx[i,j,1:]]=s

    return ret
            



def index_add(a,belong,N):
    b,h,n,d=a.size()
    device=a.device
    torch.cuda.synchronize()
    st=time.time()
    group=torch.zeros(b,h,N,d,device=device)
    torch.cuda.synchronize()
    speed.idx_add_zeros+=time.time()-st
    
    torch.cuda.synchronize()
    st=time.time()
    for i in range(b):
        for j in range(h):
            group[i,j]=torch.index_add(group[i][j],0,belong[i][j],a[i,j])
            #group[i,j].index_add_(0,belong[i][j],a[i,j])
    torch.cuda.synchronize()
    speed.idx_add_idxadd+=time.time()-st

    group=group.float()
                
    return group



def set_all_seed(opt):
    seed=opt.all_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_data_seed(opt):
    seed=opt.data_seed
    np.random.seed(seed)
    random.seed(seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        shutil.rmtree(checkpoint)



def kmeans_init(x,N,init):
    b,h,n,d=x.size()
    device=x.device
    c=None
    if(init=='rand'):
        idx=torch.randperm(n,device=device)[:N]
        c=x[:,:,idx,:]
    elif(init=='first'):
        c=x[:,:,:N,:]
    elif(init=='kmeans++'):
        c=np.zeros((b,h,N,d))
        for i in range(b):
            for j in range(h):
                X=x[i,j].cpu().numpy()
                n_samples = X.shape[0]
                random_state = check_random_state(None)
                x_squared_norms = row_norms(X, squared=True)
                c[i,j]=_k_init(X,N,x_squared_norms=x_squared_norms,random_state=random_state)
        c=torch.from_numpy(c).to(device)
    elif(init=='my'):
        c=my_k_init(x,N)
    return c












def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

def my_k_init(x,N):
    b,h,n,d=x.size()
    device=x.device
    dis=torch.zeros(b,h,n,device=device)
    c=torch.zeros(b,h,N,d,device=device)

    a2=torch.square(x).sum(-1)

    def get_new(a2,c):
        c=c.unsqueeze(2)
        b2=torch.square(c).sum(-1)
        
        #new_dis_t=((x-c)**2).sum(-1)
        ab=torch.matmul(x,c.transpose(-1,-2)).squeeze(-1)
        new_dis=a2+b2-2*ab
        return new_dis

    idx=random.randint(0,n-1)
    dis=get_new(a2,x[:,:,idx])
    c[:,:,0]=x[:,:,idx]

    bt=torch.arange(b).unsqueeze(1).unsqueeze(2).to(device)
    ht=torch.arange(h).unsqueeze(0).unsqueeze(2).to(device)

    for i in range(1,N):
        idx=torch.argmax(dis,-1)
        new_c=x[bt,ht,idx.unsqueeze(2)].squeeze(2)
        c[:,:,i]=new_c
        new_dis=get_new(a2,new_c)
        dis=(dis+new_dis-torch.abs(dis-new_dis))/2
    
    return c
        
        


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def output_metrics(args,cft):
    n=args.num_class
    right,tot=0,0
    for i in range(n):
        right+=cft[i][i]
        for j in range(n):
            print("%6d"%(cft[i][j]),end='')
            tot+=cft[i][j]
        print()
    
    F1=[0 for i in range(n)]
    for i in range(n):
        fz=2*cft[i][i]
        fm=0
        for j in range(n):
            fm+=cft[i][j]+cft[j][i]
        F1[i]=fz/fm
    macroF1=0
    for i in range(n):
        macroF1+=F1[i]
    macroF1/=n

    print('%-20s %lf'%('micro-F1(accuracy):',right/tot))
    print('%-20s %lf'%('macro-F1:',macroF1))

    for i in range(len(glo.keep_rate_list)):
        a=glo.keep_rate_list[i]
        n=len(a)//4
        a.sort()
        print(a[n],a[2*n],a[3*n])







def my_save(ckp_prefix,model,args,optimizer,scheduler,epoch_done,
            for_resume,rotate,global_sample,last_log,last_dev,last_N,
            training_time):
    if(for_resume==True):
        output_dir = os.path.join(args.output_dir, "{}-{}".format(ckp_prefix, epoch_done))
        os.makedirs(output_dir, exist_ok=True)

        print("Saving model checkpoint to", output_dir)
        state={'net':model.state_dict()}
        torch.save(state,os.path.join(output_dir, "para.bin"))

        print("Saving model args to", output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        print("Saving global_sample to", output_dir)
        torch.save(global_sample,os.path.join(output_dir, "global_sample.bin"))

        print("Saving last_log to", output_dir)
        torch.save(last_log,os.path.join(output_dir, "last_log.bin"))

        print("Saving last_dev to", output_dir)
        torch.save(last_dev,os.path.join(output_dir, "last_dev.bin"))

        print("Saving last_N to", output_dir)
        torch.save(last_N,os.path.join(output_dir, "last_N.bin"))

        print("Saving N to", output_dir)
        torch.save(glo.N,os.path.join(output_dir, "N.bin"))

        print("Saving keep-rate/merge/split list to", output_dir)
        torch.save(glo.keep_rate_list,os.path.join(output_dir, "kprate_list.bin"))
        torch.save(glo.merge_list,os.path.join(output_dir, "merge_list.bin"))
        torch.save(glo.split_list,os.path.join(output_dir, "split_list.bin"))

        print("Saving training time to", output_dir)
        torch.save(training_time,os.path.join(output_dir, "training_time.bin"))

        print("Saving warming to", output_dir)
        torch.save(glo.warming,os.path.join(output_dir, "warming.bin"))

        print("Saving optimizer and scheduler states to", output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    else:
        print('Saving for no resume')
        output_dir=os.path.join(args.output_dir, ckp_prefix)
        if(global_sample!=None):
            output_dir = os.path.join(args.output_dir, "{}-{}".format(ckp_prefix, global_sample))
        os.makedirs(output_dir, exist_ok=True)
        
        state={'net':model.state_dict()}
        torch.save(state,os.path.join(output_dir, "para.bin"))

    if(rotate==True):
        _rotate_checkpoints(args, ckp_prefix)




def my_restore(args,model,optimizer,scheduler):
    st_epoch,st_global,st_log,st_dev,st_N,training_time=0,0,0,0,0,0

    if(args.restore_model!='None'):
        print('Restoring model-----',args.restore_model)
        dir=args.restore_model
        path=dir+'/para.bin'
        state_dict=torch.load(path)['net']
        model.load_state_dict(state_dict)
        

        if(args.resume_training==True):
            glo.N=torch.load(dir+'/N.bin')
            state_dict=torch.load(dir+'/optimizer.pt')
            optimizer.load_state_dict(state_dict)

            state_dict=torch.load(dir+'/scheduler.pt')
            scheduler.load_state_dict(state_dict)
    
            st_global=int(torch.load(dir+'/global_sample.bin'))
            st_log=int(torch.load(dir+'/last_log.bin'))
            st_dev=int(torch.load(dir+'/last_dev.bin'))
            st_N=int(torch.load(dir+'/last_N.bin'))
            st_epoch=int(dir[dir.find('-')+1:])+1

            glo.keep_rate_list=torch.load(dir+'/kprate_list.bin')
            glo.split_list=torch.load(dir+'/split_list.bin')
            glo.merge_list=torch.load(dir+'/merge_list.bin')

            glo.warming=torch.load(dir+'/warming.bin')
            training_time=torch.load(dir+'/training_time.bin')
    else:
        if(args.resume_training==True):
            raise ValueError('resume=true restore=None')
        else:
            print('Begin training from scratch')
    
    return model,optimizer,scheduler,st_epoch,st_global,st_log,st_dev,st_N,training_time



def group_measure(a,compatness=None,query=None):
    measurement=glo.args.N_measurement
    device=a.device
    if(glo.N_policy!='auto'):
        return
    if(measurement!='cos' and measurement!='L2'):
        return
    a=a.detach()
    b,h,N,d=a.size()    

    inf=99999

    if(measurement=='cos'):
        dis,_=group_distance(a,measurement='cos')
        threshold=glo.args.N_cos_thre
        mask=torch.ones(N,N,device=device).triu()*inf
        down=dis-dis.triu()-mask
        mx=down.max(-1).values
        cnt=(mx<threshold).int()
        keep=cnt.sum(-1).view(-1)
        keep_rate=keep/N
        glo.keep_rate_list[glo.layer]+=keep_rate.cpu().tolist()
    else:
        dis2,_=group_distance(a,measurement='L2')
        a2=cal_len2(query)
        R2=a2.mean(dim=-1)
        R2_inv=1/R2
        
        eps=glo.args.N_L2_eps
        d2=(math.log(eps)**2)/4*R2_inv
        split_thre=d2
        merge_thre=4*d2

        mask=torch.ones(N,N,device=device).triu()*inf
        down=dis2-dis2.triu()+mask
        mi=down.min(-1).values
        merge_N=(mi<merge_thre.unsqueeze(-1)).int()
        split_N=(compatness>split_thre.unsqueeze(-1)).int()

        split=split_N.sum(-1).view(-1)
        merge=merge_N.sum(-1).view(-1)
        glo.split_list[glo.layer]+=split.cpu().tolist()
        glo.merge_list[glo.layer]+=merge.cpu().tolist()

        



    

def update_N_auto_once(args,global_sample,writer):

    for i in range(args.num_layers):
        a=args.N_a
        oldN=glo.N[i]
        goal=args.N_kprate_lwb
        newN=None

        if(args.N_measurement=='cos'):
            if(kprate>=args.N_kprate_lwb or kprate<=args.N_kprate_upb):
                continue
            list=glo.keep_rate_list[i]
            glo.keep_rate_list[i]=[]
            list.sort()
            pos=int(len(list)*0.75)
            kprate=list[pos]
            newN=oldN*kprate/goal
            print(kprate)
            writer.add_scalar('kprate'+str(i),kprate,global_sample)
        else:
            split_list,merge_list=glo.split_list[i],glo.merge_list[i]
            glo.split_list[i],glo.merge_list[i]=[],[]
            split_sum,merge_sum=sum(split_list),sum(merge_list)
            #split=split_sum/len(split_list)
            merge=merge_sum/max(1,len(merge_list))
            dlt=-merge
            newN=oldN+dlt
            writer.add_scalar('merge'+str(i),merge,global_sample)

        N=a*oldN + (1-a)*newN
        N=max(N,glo.N_min)
        glo.N[i]=N
    
    for i in range(args.num_layers):
        writer.add_scalar('N'+str(i),glo.N[i],global_sample)



def update_warming_N(args,global_sample,last_N,now_epoch,writer):
    warmup=args.N_warmup
    new_last_N=last_N
    if(glo.warming==True):
        if(global_sample>=warmup):
            glo.warming=False
            for i in range(args.num_layers):
                glo.N[i]=args.N_init
                glo.keep_rate_list.append([])
                glo.split_list.append([])
                glo.merge_list.append([])
            for i in range(args.num_layers):
                writer.add_scalar('N'+str(i),glo.N[i],global_sample)
    else:
        if((global_sample-last_N)>args.N_sample):
            if(args.N_policy=='manual'):
                mxN=0
                for i in range(len(args.N_list)):
                    if(args.N_list[i][0]<=now_epoch):
                        mxN=args.N_list[i][1]
                for i in range(args.num_layers):
                    glo.N[i]=mxN
            else:
                update_N_auto_once(args,global_sample,writer)
            new_last_N=global_sample
    
    return new_last_N



def my_linear_schedule(optimizer,num_training_steps,last_epoch=-1):
    def lr_lambda(current_step: int):
        linear_lbd=max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))
        bsz_lbd=glo.batch_size**0.5
        return linear_lbd*bsz_lbd

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def upd_rec(rec,dlt):
    ret={}
    for k in dlt: 
        ret[k]=[0,0]
        if(rec.get(k)!=None):
            ret[k][0],ret[k][1]=rec[k][0],rec[k][1]
        ret[k][0]+=dlt[k][0]
        ret[k][1]+=dlt[k][1]
    return ret


def write_rec(rec,writer,global_sample,name):
    for k in rec:
        out_name=name+'_'+k
        out_d=1.0*rec[k][0]/rec[k][1]
        writer.add_scalar(out_name,out_d,global_sample)



def process_bar(percent):
    bar = '\r' + ' {:0>4.1f}%'.format(percent*100) 
    print(bar, end='', flush=True)
    if(percent>0.999):
        print()


def upd_right(pred,label,mask,args):
    ret={}
    if(args.task=='for'):
        donothing=0
    elif(args.task=='cls'):
        n,m=pred.size(0),pred.size(1)
        dr=0
        for i in range(n):
            mx=0
            for j in range(m):
                if(pred[i,j]>pred[i,mx]):
                    mx=j
            glo.res[label[i].int()][mx]+=1
            if(mx==label[i]):
                dr+=1
        dt=n
        ret['acc']=(dr,dt)
    elif(args.task=='imp'):
        b,n,d=pred.size()
        err=torch.abs(pred-label)
        lbd_list=[0.1,0.01,0.001]
        mask=mask.unsqueeze(-1)
        tot=mask.sum()*d
        for lbd in lbd_list:
            abs_bound=lbd
            rel_bound=torch.abs(lbd*label)

            abs_right=((err<abs_bound).int()*mask).sum()
            rel_right=((err<rel_bound).int()*mask).sum()
            ret['abs_acc_'+str(lbd)]=[abs_right,tot]
            ret['rel_acc_'+str(lbd)]=[rel_right,tot]
        
    return ret

def fitting(n):
    return glo.fiter.fit(n,True)



def cal_loss(pred,label,device,args,mask=None):
    loss=None
    logits=pred.to(device)
    gt=label.to(device)

    if(args.task=='cls'):
        gt=gt.long()
        loss=F.cross_entropy(logits,gt,reduction='mean')
    elif(args.task=='for'):
        loss=F.mse_loss(logits,gt,reduction='mean')
    elif(args.task=='imp'):
        _,_,d=logits.size()
        s2=F.mse_loss(logits,gt,reduction='none')
        s2,cnt=(s2*mask.unsqueeze(-1)).sum(),mask.sum()*d
        loss=s2/cnt

    return loss

import pynvml

def get_cuda(args):
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.cuda_idx)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    use_rate=1.0*meminfo.used/meminfo.total
    return use_rate

def clean_cuda(args):
    while(1):
        torch.cuda.empty_cache()
        t=get_cuda(args)
        if(t<0.3):
            break


def fetch_few_data(dataset,args):
    print('fetching',args.few,'samples per class')
    ret=[]
    cnt=[0 for _ in range(args.num_class)]
    patch=args.few*args.num_class
    st=0
    input,label,mask=[],[],[]
    done=0
    while(done<args.num_class):
        ed=st+patch
        batch=dataset.fetch(st,ed,args)
        for i in range(patch):
            lab=int(batch['label'][i])
            if(cnt[lab]==args.few):
                continue
            cnt[lab]+=1
            if(cnt[lab]==args.few):
                done+=1
            input.append(batch['input'][i].tolist())
            label.append(batch['label'][i].tolist())
            mask.append(batch['mask'][i].tolist())
        st=ed
    
    ret=make_dataset_from_tensor(input,label,mask,args)
    return ret
            





def get_default_para(args):
    num_class,num_channel,seg_len,task=None,None,None,None
    lr,epoch=None,None
    pre,train,dev,few=None,None,None,None
    N_policy,N_L2_eps,N_list,N_init=None,None,None,None
    batch_size=None

    if(args.mode=='pretrain'):
        pre,train,dev,few=0.9,0.,0.1,None
        lr,epoch=5e-5,100
        task='imp'
    else:
        pre,train,dev=0.,0.9,0.1
        lr,epoch=1e-4,100
        task='cls'
        if(args.data_size=='few'):
            few,epoch=100,50

    if(args.dataset=='wisdm'):
        args.datafile='accel_watch'
        num_class,num_channel,seg_len=18,3,200
        batch_size=256
        if(args.data_size=='full'):
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,32],32
        else:
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,64],64
    elif(args.dataset=='hhar'):
        args.datafile='hhar'
        num_class,num_channel,seg_len=5,3,200
        batch_size=256
        if(args.data_size=='full'):
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,32],32
        else:
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,64],64
    elif(args.dataset=='rwhar'):
        args.datafile='rwhar_scale'
        num_class,num_channel,seg_len=8,3,200
        batch_size=256
        if(args.data_size=='full'):
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,32],32
        else:
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,64],64
    elif(args.dataset=='ecg'):
        args.datafile='ecg'
        if(args.mode=='pretrain'):
            batch_size=128
            num_class,num_channel,seg_len=9,12,400
            N_policy,N_L2_eps,N_list,N_init='manual',None,[0,64],64
        else:
            if(args.data_size=='full'):
                num_class,num_channel,seg_len=9,12,2000
                N_policy,N_L2_eps,N_list,N_init='auto',2,None,None
            else:
                batch_size=128
                num_class,num_channel,seg_len=9,12,400
                N_policy,N_L2_eps,N_list,N_init='manual',None,[0,64],64

    N_init=check_None(N_init,seg_len//10)

    args.num_class=check_None(args.num_class,num_class)
    args.num_channel=check_None(args.num_channel,num_channel)
    args.seg_len=check_None(args.seg_len,seg_len)
    args.task=check_None(args.task,task)
    args.epoch=check_None(args.epoch,epoch)
    args.lr=check_None(args.lr,lr)
    args.pre=check_None(args.pre,pre)
    args.train=check_None(args.train,train)
    args.dev=check_None(args.dev,dev)
    args.few=check_None(args.few,few)
    args.N_policy=check_None(args.N_policy,N_policy)
    args.N_L2_eps=check_None(args.N_L2_eps,N_L2_eps)
    args.N_list=check_None(args.N_list,N_list)
    args.N_init=check_None(args.N_init,N_init)
    args.batch_size=check_None(args.batch_size,batch_size)

    glo.res=[[0 for _ in range(args.num_class)] for _ in range(args.num_class)]

    return args

def check_None(a,b):
    if(a==None):
        return b
    return a



def make_log_name(args):
        tr=str(args.train) if(args.few==None) else str(args.few)
        s=str(args.pre)+'-'+tr+'-'+str(args.dev)
        t=str(args.pre_overlap)+'-'+str(args.data_overlap)+'-'+args.data_method
        v='seed'+str(args.all_seed)
        w=('ksiz'+str(args.ksiz)+'_stride'+str(args.stride)+'_pad-'+str(args.pad)+
            '-'+args.prj)
        task=args.task+(
            (args.imp_method+str(args.mask_rate) if(args.task=='imp') else '')
            )

        dataset=args.dataset+(str(args.seg_len) if(args.cut_len==None) else str(args.cut_len))
        args.log_name=args.mode+'-'+task+'_'+dataset+'_'+t+'_'+s+'_'+w+'_'+v
        
        model_name='_'+args.model

        if(args.model=='BERT'):
            model_name='_GA-'+args.N_policy+'-'
            if(args.N_policy=='manual'):
                model_name+=str(args.N_list)
            else:
                model_name+=(args.N_measurement+'-'+
                                    (str(args.N_cos_thre) if(args.N_measurement=='cos') else str(args.N_L2_eps))+
                                    '-min'+str(args.N_min)+
                                    '-kprate'+str(args.N_kprate_upb)+str(args.N_kprate_lwb)+
                                    '-alpha'+str(args.N_a)+
                                    '-Kmeans'+str(args.Kmeans_rounds))
        
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

        if(args.resume_training==True):
            args.log_name+='_resume'
        
        args.log_name=args.log_name.replace(' ','')
        
        return args


