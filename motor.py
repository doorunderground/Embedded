import serial
import time

PORT = 'COM5'
BAUDRATE = 9600

COMMANDS = {
    'a': 'Left Drive',
    'd': 'Right Drive',
    's': 'Stop',
}

ser = serial.Serial(PORT, BAUDRATE, timeout=1)
print(f"✅ {PORT} 포트 연결 성공! (속도: {BAUDRATE})")
print("조작 방법: a=왼쪽  d=오른쪽  s=정지  q=종료")
print("-" * 40)

while True:
    key = input('> ').strip().lower()

    if key == 'q':
        # 종료 전 정지 명령 전송
        ser.write(b's\n')
        print("종료합니다.")
        break

    if key not in COMMANDS:
        print(f"  알 수 없는 키: '{key}'  (a / d / s / q 만 사용)")
        continue

    # 전송
    msg = key + '\n'
    ser.write(msg.encode())
    print(f"  [송신] {key}  →  {COMMANDS[key]}")

    # 수신 (아두이노 응답 대기, 최대 2줄)
    for _ in range(2):
        raw = ser.readline()
        if not raw:
            break
        text = raw.decode('utf-8', errors='replace').strip()
        if text:
            print(f"  [수신] {text}")

    time.sleep(0.05)

ser.close()