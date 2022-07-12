import torch
import torch.nn as nn

in_c=10
out_c=8
sz=5
stride=5
padding=sz//2


input=torch.randn(4,in_c,10000)



conv=nn.Conv1d(in_channels=in_c,out_channels=out_c,kernel_size=sz,
                    stride=stride,padding=0,padding_mode='circular')

tconv=nn.ConvTranspose1d(in_channels=out_c,out_channels=in_c,kernel_size=sz,
                    stride=stride,padding=0)

output=conv(input)
proj=tconv(output)

print(input.size())
print(output.size())
print(proj.size())