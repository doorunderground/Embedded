# %% [0] 라이브러리 & 모델 로드
import numpy as np
import joblib
import torch
import torch.nn as nn
import serial
import serial.tools.list_ports
import time

# ── 신경망 구조 (2_train_models.py 와 동일) ──────────────
class CalibNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

# 모델 로드
MODELS_DIR = "C:\\stm32_project\\ultrasonic_calibration\\models"
lin  = joblib.load(f"{MODELS_DIR}\\linear_model.pkl")
poly = joblib.load(f"{MODELS_DIR}\\poly_model.pkl")

ckpt   = torch.load(f"{MODELS_DIR}\\nn_model.pth", map_location="cpu")
nn_model = CalibNet()
nn_model.load_state_dict(ckpt["state_dict"])
nn_model.eval()
X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

def predict_nn(x_cm):
    x_norm = (x_cm - X_mean) / X_std
    with torch.no_grad():
        y_norm = nn_model(torch.FloatTensor([[x_norm]])).item()
    return y_norm * y_std + y_mean

print("모델 로드 완료!")
print("사용 가능한 포트:", [p.device for p in serial.tools.list_ports.comports()])

# %% [1] 시리얼 포트 설정
PORT = "COM3"   # ← 본인 포트로 변경 (위에서 확인)
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(0.5)
print(f"연결됨: {PORT} @ {BAUD}baud")

# %% [2] 실시간 측정 (Ctrl+C 로 중단)
import re

PATTERN = re.compile(r"\((\d+\.\d+)\s*cm\)")

print(f"\n{'원시(cm)':>10} │ {'선형':>10} │ {'다항2차':>10} │ {'신경망':>10} │ 오차(nn)")
print("─" * 65)

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        m = PATTERN.search(line)
        if not m:
            continue

        raw = float(m.group(1))
        c_lin  = lin.predict([[raw]])[0]
        c_poly = poly.predict([[raw]])[0]
        c_nn   = predict_nn(raw)
        err    = c_nn - raw

        print(f"{raw:>10.2f} │ {c_lin:>10.2f} │ {c_poly:>10.2f} │ {c_nn:>10.2f} │ {err:>+.2f} cm")

except KeyboardInterrupt:
    ser.close()
    print("\n연결 종료.")

# %% [3] 연결 종료 (셀 [2] 강제 중단 후 실행)
try:
    ser.close()
    print("포트 닫힘.")
except:
    print("이미 닫혀 있습니다.")