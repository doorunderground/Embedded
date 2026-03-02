#%%

import pandas as pd

df = pd.read_csv('C:\\stm32_project\\ultrasonic_test\\supersonic.csv')
df.head(5)






# %%
#
#
############################ [1]

import numpy as np
x = df['measure'].values.reshape(-1, 1).astype(np.float32)
y = df[' true'].values.reshape(-1, 1).astype(np.float32)

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
import matplotlib.pyplot as plt
plt.plot(x, y, '.')
plt.plot(x, pred, '.')
plt.show()






# %%
