import torch
# %%
x = torch.arange(4)
A = torch.arange(20).reshape(5, 4)
B = torch.tensor(torch.ones(4, 3), dtype=torch.long)
x, A, B

# %%
A.shape, x.shape, torch.mv(A, x)  # 矩阵向量乘法

# %%
torch.mm(A, B)

# %%
A / A.sum(axis=1, keepdims=True)
