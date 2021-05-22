import torch
import torchcsr

crow_indices = [0, 2, 3]
col_indices = [0, 1, 0]
values = [1, 2, 3]
a = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int32),
                        torch.tensor(col_indices, dtype=torch.int32),
                        torch.tensor(values), dtype=torch.float)
b = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int32),
                        torch.tensor(col_indices, dtype=torch.int32),
                        torch.tensor(values), dtype=torch.float)

def fake_softmax(a, dim):
    assert dim == 1
    padded = a.to_dense()
    mask = padded == 0
    padded.masked_fill_(mask, float("-inf"))
    return torch.softmax(padded, 1).to_sparse_csr()

res_a = fake_softmax(a, 1)
res_b = torch.softmax(b, 1)
print(res_a)
print(res_b)
print(torch.allclose(res_a.to_dense(), res_b.to_dense()))
