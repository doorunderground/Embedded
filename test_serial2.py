import serial
from datetime import datetime

PORT = 'COM3'
BAUDRATE = 9600

try:
    with serial.Serial(PORT, BAUDRATE, timeout=6) as ser:
        print(f"[연결] {PORT} ({BAUDRATE} baud) - 종료: Ctrl+C")
        print("HC-SR04 거리 측정 시작...\n")

        while True:
            raw = ser.readline()
            text = raw.decode('utf-8', errors='replace').strip()
            if text:
                now = datetime.now().strftime("%H:%M:%S")
                try:
                    dist = float(text)
                    print(f"[{now}] 거리: {dist:.2f} cm")
                except ValueError:
                    print(f"[{now}] {text}")

except serial.SerialException as e:
    print(f"[오류] 포트 연결 실패: {e}")
except KeyboardInterrupt:
    print("\n[종료]")
