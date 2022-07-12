from train import train


def pretrain(dataset_pretrain,dataset_dev,args,model,writer,device):
    train(dataset_pretrain,dataset_dev,args,model,writer,device)