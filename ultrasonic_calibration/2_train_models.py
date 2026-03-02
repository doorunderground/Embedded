# %% [0] 라이브러리 임포트
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

print("라이브러리 로드 완료")

# %% 
# 
# 
# [1] 데이터 로드

DATA_FILE = "C:\\stm32_project\\ultrasonic_calibration\\data.csv"
df = pd.read_csv(DATA_FILE)
X_raw = df["measured_cm"].values.astype(np.float32)
y_raw = df["actual_cm"].values.astype(np.float32)
X = X_raw.reshape(-1, 1)
# sklearn 모델은 2차원 배열을 요구함
# X_raw = [5.3, 13.05, 14.0, 19.8 ...]  shape: (6, ) -> 1차원
# X = (6, 1) -> 2차원

print("=" * 50)
print("  로드된 데이터")
print("=" * 50)
print(f"{'측정(cm)':>10}  {'실측(cm)':>10}  {'오차(cm)':>10}")
print("-" * 35)
for m, a in zip(X_raw, y_raw):
    print(f"{m:>10.1f}  {a:>10.1f}  {a - m:>+10.1f}")

#%%

# %% 
# 
# 
# [2] 선형 회귀 (Linear Regression)

lin = LinearRegression()
lin.fit(X, y_raw)
# X(측정값), y_raw(실측값)을 보고 최적의 a, b 찾기
# f(x) = ax + b
# 오차^2 의 합이 최소가 되는 a, b를 계산

y_lin = lin.predict(X)
# X = [5.3, 13.05, 14.0, 19.8, 27.19]
# y_lin = [f(5.3), f(13.05), f(14.0), f(19.8)]

rmse_lin = np.sqrt(mean_squared_error(y_raw, y_lin))
# RMSE = 예측이 얼마나 틀렸는지 수치화
# MSE = 평균((실측 - 예측)²)
# RMSE = √MSE 

print(f"[1] 선형 회귀")
print(f"    f(x) = {lin.coef_[0]:.4f}·x + {lin.intercept_:.4f}")
#                     기울기 a                y절편 b
print(f"    RMSE = {rmse_lin:.4f} cm")


#%%


import matplotlib.pyplot as plt
plt.plot(X, y_raw, '.')
plt.plot(X, y_lin, '.')
plt.show()



# %% 
# 
# 
# [3] 다항 회귀 2차 (Polynomial Regression)

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

poly.fit(X, y_raw)
y_poly = poly.predict(X)
rmse_poly = np.sqrt(mean_squared_error(y_raw, y_poly))

coefs = poly.named_steps["lin"].coef_
inter = poly.named_steps["lin"].intercept_
print(f"[2] 다항 회귀 (2차)")
print(f"    f(x) = {coefs[2]:.4f}·x² + {coefs[1]:.4f}·x + {inter:.4f}")
print(f"    RMSE = {rmse_poly:.4f} cm")




# %% [4] 신경망 정의 및 데이터 정규화 (Neural Network)
# 입출력 표준화 (평균 0, 표준편차 1)
X_mean, X_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

X_norm = (X_raw - X_mean) / X_std
y_norm = (y_raw - y_mean) / y_std

X_t = torch.FloatTensor(X_norm).unsqueeze(1)
y_t = torch.FloatTensor(y_norm).unsqueeze(1)

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

    def forward(self, x):
        return self.net(x)

print(f"정규화 완료  X: mean={X_mean:.2f}, std={X_std:.2f}")
print(f"            y: mean={y_mean:.2f}, std={y_std:.2f}")
print(CalibNet())

# %% [5] 신경망 학습
torch.manual_seed(42)
model = CalibNet()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
criterion = nn.MSELoss()

losses = []
EPOCHS = 8000
for epoch in range(EPOCHS):
    model.train()
    pred = model(X_t)
    loss = criterion(pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 1000 == 0:
        print(f"    epoch {epoch+1:>5} | loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_nn_norm = model(X_t).numpy().flatten()
y_nn = y_nn_norm * y_std + y_mean
rmse_nn = np.sqrt(mean_squared_error(y_raw, y_nn))

print(f"\n[3] 신경망 (MLP 1→32→32→1)")
print(f"    RMSE = {rmse_nn:.4f} cm")

# %% [6] 결과 비교표
print(f"\n{'측정(cm)':>10}  {'실측(cm)':>10}  {'선형':>8}  {'다항2차':>8}  {'신경망':>8}")
print("-" * 55)
for m, a, l, p, n in zip(X_raw, y_raw, y_lin, y_poly, y_nn):
    print(f"{m:>10.1f}  {a:>10.1f}  {l:>8.2f}  {p:>8.2f}  {n:>8.2f}")
print(f"{'RMSE':>21}  {rmse_lin:>8.4f}  {rmse_poly:>8.4f}  {rmse_nn:>8.4f}")

# %% [7] 모델 저장
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

# %% [8] 시각화
x_plot  = np.linspace(0, 35, 300).reshape(-1, 1).astype(np.float32)
xp_flat = x_plot.flatten()

xp_norm = (xp_flat - X_mean) / X_std
with torch.no_grad():
    yp_nn = model(torch.FloatTensor(xp_norm).unsqueeze(1)).numpy().flatten()
yp_nn = yp_nn * y_std + y_mean

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 보정 함수 비교
ax = axes[0]
ax.plot([0, 35], [0, 35], "k--", alpha=0.4, label="보정 없음 (y=x)")
ax.plot(xp_flat, lin.predict(x_plot),  label=f"선형     RMSE={rmse_lin:.2f}cm")
ax.plot(xp_flat, poly.predict(x_plot), label=f"다항2차  RMSE={rmse_poly:.2f}cm")
ax.plot(xp_flat, yp_nn,                label=f"신경망   RMSE={rmse_nn:.2f}cm")
ax.scatter(X_raw, y_raw, s=120, zorder=5, color="red", label="데이터")
ax.set_xlabel("측정 거리 (cm)")
ax.set_ylabel("보정 후 거리 (cm)")
ax.set_title("보정 함수 f(측정값) = 실측값")
ax.legend()
ax.grid(True, alpha=0.3)

# 오른쪽: 학습 손실 곡선
ax2 = axes[1]
ax2.plot(losses, color="steelblue")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (MSE, 정규화 공간)")
ax2.set_title("신경망 학습 손실 곡선")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("calibration_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("그래프 저장: calibration_result.png")