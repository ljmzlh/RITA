import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, dataloader
from tqdm import tqdm
import numpy as np
import os
import time
from utils import glo,speed,output_speed
import json
import random
from utils import output_metrics,my_restore
from train import dev_epoch

global_step,acc_loss=0,0
dev_min=1e9
dev_acc=0

cft=None

        

def dev(dataset_dev,args,model,writer,device):
    model,_,_,_,_,_,_,_=my_restore(args,model,None,None)
    dev_epoch(args,dataset_dev,model,writer)