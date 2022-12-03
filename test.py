import numpy as np
import torch


#batch 2
#dim 3
#[2, neg_target, 3]
a = torch.tensor([[1,2,3],[4,5,6]]).cuda() #[2,3]
print(a)

b = a.unsqueeze(1).expand([2,2,3])

c = b

print(b[0])

s = -9e10*torch.ones((2,3))
print(s)

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())