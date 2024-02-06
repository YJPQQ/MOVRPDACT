import torch
import numpy as np


pref = torch.rand((2))
pref = pref / torch.sum(pref)
print(pref)

concat_pref = pref[None, None, :] .expand(2, 20, -1) 
print(concat_pref)
print(concat_pref.shape)

data = torch.zeros(size=(2,20,1))
print(data)
print(data.shape)

data = torch.cat((data, concat_pref), dim = 2)
print(data)
print(data.shape)
