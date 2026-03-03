#%%

# 
#
# ################################# [IMPORT]

import torch
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\stm32_project\\ultrasonic_torch\\supersonic.csv')

x = df['measure']
y = df[' true']

x = torch.tensor(x).reshape(-1, 1).float()
y = torch.tensor(y).reshape(-1, 1).float() * 1000
# Numpy / Pandas -> astype(np.float32)
# PyTorch -> .float()

plt.plot(x, y, '.')


# %%

# 
#
# ################################# 

class Preproc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return (x - 20000) / 10000

class PostProc(torch.nn.Module):
    def forward(self, x):
        return x * 9900 + 21000

# 1 -> 4 -> 1
model = torch.nn.Sequential(
            Preproc(),
            torch.nn.Linear(1, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1),
            PostProc()
)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        a = torch.randn(())
        self.a = torch.nn.Parameter(a)
        b = torch.randn(())
        self.b = torch.nn.Parameter(b)
        c = torch.randn(())
        self.c = torch.nn.Parameter(c)
        
    def forward(self, x):
        return self.a * x ** 2 + self.b * x + self.c

model = MyModel()

opt = torch.optim.Adam(model. parameters())
loss_fn = torch.nn.MSELoss()

for epoch in range(50000):
    # fedd forward
    pred = model(x)
    # loss
    loss = loss_fn(pred, y)
    # grad
    opt.zero_grad()
    loss.backward()
    # update
    opt.step()
    print(f'{epoch}  {loss.item()}\r', end='')
    




# %%

# 
#
# ################################# 

with torch.no_grad():
    pred = model(x)

plt.plot(x, y, '.')
plt.plot(x, pred, '.')
plt.axis('off')
plt.show()
# %%
