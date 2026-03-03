# %% [0] 
# 
#
# ################################# [IMPORT]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

print("IMPORT COMPLETE")





# %% 
# 
# 
# ################################# [1][DATA LOAD]

df = pd.read_csv('C:\\stm32_project\\ultrasonic_calibration\\data.csv')
x = df["measured_cm"].values.reshape(-1, 1).astype(np.float32)
y = df["actual_cm"].values.astype(np.float32)

# sklearn 모델은 2차원 배열을 요구함
# x = [5.3, 13.05, 14.0, 19.8 ...]  shape: (6, ) -> 1차원
# x = (6, 1) -> 2차원

print("=" * 50)
print(f"{'측정(cm)':>10}  {'실측(cm)':>10}  {'오차(cm)':>10}")
print("-" * 50)
for m, a in zip(x, y):
    print(f"{m[0]:>10.1f}  {a:>10.1f}  {a - m[0]:>+10.1f}")





# %% 
# 
# 
# ################################# [2] 선형 회귀 (Linear Regression)

lin = LinearRegression()
lin.fit(x, y)
# x(측정값), y(실측값)을 보고 최적의 a, b 찾기
# f(x) = ax + b
# 오차² 의 합이 최소가 되는 a, b를 계산

y_lin = lin.predict(x)
# x = [5.3, 13.05, 14.0, 19.8, 27.19]
# y = [f(5.3), f(13.05), f(14.0), f(19.8)]

rmse_lin = np.sqrt(mean_squared_error(y, y_lin))
# RMSE = 예측이 얼마나 틀렸는지 수치화
# MSE = 평균((실측 - 예측)²)
# RMSE = √MSE 

print(f"[1] 선형 회귀")
print(f"    f(x) = {lin.coef_[0]:.4f}·x + {lin.intercept_:.4f}")
#                     기울기 a                y절편 b
print(f"    RMSE = {rmse_lin:.4f} cm")

#RMSE = 0.85cm -> 예측이 실측과 평균 0.85cm 차이난다.




#%%
#
#
# ################################# 선형 회귀 - 실제 데이터 vs 예측 값

import matplotlib.pyplot as plt
plt.plot(x, y, '.')        # 파란 점 - 실제 데이터
plt.plot(x, y_lin, 'o')    # 주황 점 - 선형 회귀 예측 값
plt.axis('off')
plt.show()




# %% 
# 
# 
################################# [3] 다항 회귀 2차 (Polynomial Regressin)

poly = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=True)),
    ("lin",  LinearRegression()),
])

# PolynomialFeatures
# input  -  [14.0]
# output -  [ 1,   14.0,  196.0]
#           bias    x      x²

# LinearRegression
# [1, x, x²] 를 feature로 받아서,
# y = coef[0]·1 + coef[1]·x + coef[2]·x²

poly.fit(x, y)
y_poly = poly.predict(x)
rmse_poly = np.sqrt(mean_squared_error(y, y_poly))

# coef_  = [coef[0],  coef[1],  coef[2]]
#            bias       x->b       x²->a
# inter_ = y절편 c
coefs = poly.named_steps["lin"].coef_
inter = poly.named_steps["lin"].intercept_

# coef_ / intercept_ -> fit() 후 모델이 학습한 값들

# coefs = poly.named_steps["lin"].coef_
# array([0.0, 1.0234, -0.0021])
#      bias항   x계수   x²계수 

# inter = poly.named_steps["lin"].intercept_
# -0.3821  ← 상수항


print(f"[2] 다항 회귀 (2차)")
print(f"    f(x) = {coefs[2]:.4f}·x² + {coefs[1]:.4f}·x + {inter:.4f}")
print(f"    RMSE = {rmse_poly:.4f} cm")




# %% 
# 
# 
# ################################# [4] 신경망 정의 및 데이터 정규화

# 정규화
X_mean, X_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()

X_norm = (x- X_mean) / X_std
y_norm = (y - y_mean) / y_std

#   x  =  [5.3, 13.05, 14.0, 19.8, 27.19]
# 평균 =  15.87
# 표춘편차 = 7.32

# X_norm = [-1.44, -0.38, -0.26, 0.54, 1.55]
# 결과값이 항상 -2 ~ +2 범위 안에 들어옴

# 정규화 전: x = 5.3 ~ 27.19 (범위가 큼)
# 정규화 후: x = -1.44 ~ 1.55 (범위가 작고 균일)

X_t = torch.FloatTensor(X_norm).reshape(-1, 1) #.unsqueeze(1)
y_t = torch.FloatTensor(y_norm).reshape(-1, 1) #.unsqueeze(1)
# 1D -> 2D     (5, ) -> (5, 1)

class CalibNet(nn.Module):
    """MLP: 1 → 32 → 32 → 1  (Tanh + Dropout)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

# Reul는 이미지 분류 같은 + 큰 데이터에 유리
# Tanh는 회귀/보정 + 소규모 데이터에 유리


    def forward(self, x):
        return self.net(x)

print(f"정규화 완료  X: mean={X_mean:.2f}, std={X_std:.2f}")
print(f"            y: mean={y_mean:.2f}, std={y_std:.2f}")
print(CalibNet())



# %%
#  
# 
################################# [5] 신경망 학습
torch.manual_seed(42)
model = CalibNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
criterion = nn.MSELoss()


losses = []
EPOCHS = 8000
for epoch in range(EPOCHS):
    model.train()                  
    pred = model(X_t)              # 1. 예측
    loss = criterion(pred, y_t)    # 2. 얼마나 틀렸나   
    optimizer.zero_grad()          # 3. gradient 초기화 / 매번 해줘야 함
    loss.backward()                # 4. 어떻게 고칠지 계산
    optimizer.step()               # 5. weight 업데이트
    
    losses.append(loss.item())     # 매 epoch의 loss값 저장. 나중에 그래프 그릴 때 쓰임
    if (epoch + 1) % 1000 == 0:
        print(f"    epoch {epoch+1:>5} | loss: {loss.item():.6f}")


model.eval()
with torch.no_grad():
    y_nn_norm = model(X_t).numpy().flatten()
y_nn = y_nn_norm * y_std + y_mean
rmse_nn = np.sqrt(mean_squared_error(y, y_nn))

print(f"\n[3] 신경망 (MLP 1→32→32→1)")
print(f"    RMSE = {rmse_nn:.4f} cm")



# %%
# 
# 
################################# [6] 결과 비교표

print(f"\n{'측정(cm)':>10}  {'실측(cm)':>10}  {'선형':>8}  {'다항2차':>8}  {'신경망':>8}")
print("-" * 65)
for m, a, l, p, n in zip(x.flatten(), y.flatten(), y_lin, y_poly, y_nn):
    print(f"{m:>10.1f}  {a:>10.1f}  {l:>13.2f}  {p:>10.2f}  {n:>10.2f}")
print(f"{'RMSE':>21}  {rmse_lin:>14.4f}  {rmse_poly:>10.4f}  {rmse_nn:>11.4f}")



# %%
# 
# 
#################################  [7] 모델 저장

BASE_DIR   = "C:\\stm32_project\\ultrasonic_calibration"
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(lin,  os.path.join(MODELS_DIR, "linear_model.pkl"))
joblib.dump(poly, os.path.join(MODELS_DIR, "poly_model.pkl"))
torch.save({
    "state_dict": model.state_dict(),
    "X_mean": float(X_mean), "X_std": float(X_std),
    "y_mean": float(y_mean), "y_std": float(y_std),
}, os.path.join(MODELS_DIR, "nn_model.pth"))

print(f"모델 저장 완료: {MODELS_DIR}")

