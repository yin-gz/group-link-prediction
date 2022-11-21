import numpy as np
import torch


#batch 2
#dim 3
#[2, neg_target, 3]
a = torch.tensor([[1,2,3],[4,5,6]]) #[2,3]

b = a.unsqueeze(1).expand([2,2,3])


print(b[0])
