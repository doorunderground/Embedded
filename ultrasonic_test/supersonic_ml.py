#%%

import pandas as pd

df = pd.read_csv('C:\\stm32_project\\ultrasonic_test\\supersonic.csv', skipinitialspace=True)
df.head(5)




# %%
#
#
############################ [1]

import numpy as np
x = df['measure'].values.reshape(-1, 1).astype(np.float32)
y = df['true'].values.reshape(-1, 1).astype(np.float32)

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x, y)
pred = lin.predict(x)
y.shape, pred.shape




# %%
#
#
############################ [2]

mse = ((y-pred) ** 2).mean()
rmse = mse ** 0.5
mse, rmse




# %%
#
#
############################ [3]
import matplotlib.pyplot as plt
plt.plot(x, y, '.')
plt.plot(x, pred, '.')
plt.show()




# %%
#
#
############################ [4]

x = df['measure'].values.reshape(-1, 1).astype(np.float32)
y = df['true'].values.reshape(-1, 1).astype(np.float32) * 1000

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x, y)
pred = lin.predict(x) / 1_000
y.shape, pred.shape

plt.plot(x, y / 1_000, '.')
plt.plot(x, pred, '.')





#%%
#
#
############################ 
lin.coef_, lin.intercept_
# 기울기      y절편




#%%

#
#
# 보정값을 위한 MSE 
x = df['measure'].values.reshape(-1, 1)
y = df['true'].values.reshape(-1, 1).astype(np.float32)

coef = (lin.coef_).astype(int)
intercept = (lin.intercept_).astype(int)

pred = (x * coef + intercept) // 1_000
pred

((pred - y / 1000) ** 2).mean() ** 0.5





# %%

#
#
#
import numpy as np
x = df['measure'].values.reshape(-1, 1).astype(np.float32)
y = df['true'].values.reshape(-1, 1).astype(np.float32)

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x, y)

lin.coef_, lin.intercept_


# ax + b ->
# ( [a * 1000]x + [b * 1000] / 1000)
#       A                B
A = (lin.coef_ * 1000).astype(int)
B = (lin.intercept_ * 1000).astype(int)
pred = (x * A + B) // 1000


# %%
