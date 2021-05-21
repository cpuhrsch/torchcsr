import torch
import torchcsr

crow_indices = [0, 2, 3]
col_indices = [0, 1, 0]
values = [1, 2, 3]
a = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
                        torch.tensor(col_indices, dtype=torch.int64),
                        torch.tensor(values), dtype=torch.double)
print(a.to_dense())
print(torch.softmax(a, 1))
