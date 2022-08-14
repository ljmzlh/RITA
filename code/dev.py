from utils import my_restore
from train import dev_epoch

global_step,acc_loss=0,0
dev_min=1e9
dev_acc=0
        
def dev(dataset_dev,args,model,writer,device):
    model,_,_,_,_,_,_,_=my_restore(args,model,None,None)
    dev_epoch(args,dataset_dev,model,writer)