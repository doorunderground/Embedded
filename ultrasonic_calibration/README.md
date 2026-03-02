# 초음파 센서 보정 미니 프로젝트

## 흐름

```
[STM32 측정] → [오차 발생] → [데이터 수집] → [ML 학습] → [실시간 보정]
```

## 설치

```bash
pip install numpy pandas matplotlib scikit-learn torch joblib pyserial
```

## 실행 순서

### Step 1 - 데이터 추가
```bash
python 1_collect_data.py
```

### Step 2 - 모델 학습
```bash
python 2_train_models.py
```

### Step 3 - 실시간 보정
```bash
python 3_realtime_calibrate.py --port COM5 --model nn
```

## 데이터 (data.csv)

| 측정(cm) | 실측(cm) | 오차 |
|----------|----------|------|
| 5.3      | 7.0      | +1.7 |
| 14.0     | 15.0     | +1.0 |
| 19.8     | 20.0     | +0.2 |
| 29.2     | 30.0     | +0.8 |