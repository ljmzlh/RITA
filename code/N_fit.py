from functools import total_ordering
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.optimize import curve_fit
from tqdm import tqdm

xx,yy,ret=None,None,None

def make_output(flist,rlist):
    output={'flist':[],'rlist':rlist}

    for i in range(len(flist)):
        f=flist[i]
        print(f)
        output['flist'].append({'func':f[0][1],'para':f[1].tolist()})

    out=open('fitting','w')
    out.write(json.dumps(output))

def N_fit(path,m,plot=False):
    global xx,yy,ret

    f=open(path,'r')
    s=f.readline()
    a=json.loads(s)
    x=a['N']
    xx=x
    y=np.array(a['bsz'])
    yy=y
    n=len(x)
    ret=[[None for _ in range(n+1)] for _ in range(n+1)]

    f=[[99999999 for _ in range(m+1)] for _ in range(n+1)]
    g,fr=[[None for _ in range(m+1)] for _ in range(n+1)],[[None for _ in range(m+1)] for _ in range(n+1)]
    

    for i in range(1,n):
        cost,func,p=minfit(0,i)
        f[i][1]=cost
        g[i][1]=(func,p)
        fr[i][1]=0
    
    for k in range(2,m+1):
        for i in tqdm(range(n)):
            for j in range(i):
                if(i-j+1<2):
                    continue
                cost,func,p=minfit(j,i)
                if(f[i][k]>f[j][k-1]+cost):
                    f[i][k]=f[j][k-1]+cost
                    g[i][k]=(func,p)
                    fr[i][k]=j


    total_cost=f[n-1][m]
    flist,rlist=[],[]
    i,k=n-1,m
    while(k>0):
        flist.append(g[i][k])
        rlist.append((x[fr[i][k]],x[i]))
        i,k=fr[i][k],k-1

    make_output(flist,rlist)
    


    

    if(plot==False):
        return



    
    plt.figure(figsize=(10,4))
    color=['r','green','purple','cyan','blue','purple','black']
    for i in range(m):
        func,p=flist[i][0],flist[i][1]
        tx=[j for j in range(rlist[i][0],rlist[i][1]+1)]
        ty=get_val(func,tx,p)
        _=plt.plot(tx,ty,color[i],label='fit plot '+str(i))

     # 也可以使用yvals=np.polyval(z1,x)
    
    plot1=plt.plot(x, y, '*',label='original')
    plt.xlabel('N')
    plt.ylabel('batch size')
    plt.legend(loc='upper right') # 指定legend的位置,读者可以自己help它的用法
    plt.title('polyfitting')
    plt.show()


def minfit(l,r):
    global xx,yy,ret
    if(ret[l][r]!=None):
        cost,retf,retp=ret[l][r][0],ret[l][r][1],ret[l][r][2]
        return cost,retf,retp
    
    x,y=xx[l:r+1],yy[l:r+1]
    func=[(func_zhi,'zhi'),(func_dui,'dui'),(func_3,'3')]
    
    cost,retf,retp=9999999,None,None
    for i in range(len(func)):
        try:
            p=get_coefficent(func[i],x,y)
        except:
            continue
        p=move_below(func[i],x,y,p)
        tmp=measure_cost(func[i],x,y,p)
        if(tmp<cost):
            cost,retf,retp=tmp,func[i],p
    ret[l][r]=[cost,retf,retp]
    return cost,retf,retp


def get_coefficent(func,x,y):
    p,_=curve_fit(func[0],x,y)
    return p

def measure_cost(func,x,y,p):
    yvals=get_val(func,x,p)
    ret=0
    for i in range(len(x)):
        ret+=y[i]-yvals[i]
    return ret

def move_below(func,x,y,p):
    yvals=get_val(func,x,p)
    dlt=0
    for i in range(len(y)):
        dlt=max(dlt,yvals[i]-y[i])
    p[-1]-=dlt
    return p






def get_val(func,x,p):
    ret=[]
    for i in range(len(x)):
        y=-1
        m=len(p)
        if(m==2):
            y=func[0](x[i],p[0],p[1])
        elif(m==3):
            y=func[0](x[i],p[0],p[1],p[2])
        elif(m==4):
            y=func[0](x[i],p[0],p[1],p[2],p[3])
        ret.append(y)
    return ret        

def func_zhi(x,a,b,c):
    return a*(x**b)+c

def func_dui(x,a,b):
    return a*np.log(x)+b

def func_3(x,a,b,c,d):
    return a*(x**3)+b*(x**2)+c*x+d


if __name__=='__main__':
    N_fit('points_8000',7,True)