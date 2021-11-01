import torch

# %%
x = torch.arange(12)

# %%
x
X = x.reshape(3, -1)

# %%
X

# %%
torch.randn(3, 4)

# %%
y = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y.shape

# %%
X
# %%
y

# %%
torch.cat((X, y), dim=0)

# %%
torch.cat((X, y), dim=1)

# %%
X - y

# %%
y[1:3]

# %%
print('id(y): ', id(y))
y = y + X
print('id(y): ', id(y))

# %%
y += X
print('id(y): ', id(y))

