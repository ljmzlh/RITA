def raw_to_input(args,data):
    print('Converting raw data to input')
    ret={}
    ret['pretrain']=f(data['pretrain'],args)
    ret['train']=f(data['train'],args)
    ret['dev']=f(data['dev'],args)
    return ret

def f(data,args):
    ret=[]
    for ins in data:
        t=trans(ins,args)
        ret.append(t)
    return ret

def trans(ins,args):
    ret={'input':[],'label':ins['label']}
    if(args.dataset=='ecg' or args.dataset=='mgh'):
        ret['input']=ins['data']
    elif(args.dataset=='stock'):
        ret['label']=[]
        for i in range(30):
            ret['input'].append([ins['data'][i]])
        for i in range(7):
            ret['label'].append([ins['label'][i]])
    else:
        for i in range(len(ins['x'])):
            ret['input'].append([ins['x'][i],ins['y'][i],ins['z'][i]])
    return ret