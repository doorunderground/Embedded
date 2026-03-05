import serial
import time
PORT = 'COM5' 
BAUDRATE = 9600

ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
print(f"{PORT} 포트 연결 성공 (속도 : {BAUDRATE})")

while True:
    msg = input("> ") + '\n'
    msg_bytes = msg.encode()
    ser.write(msg_bytes)
    print(f'[송신] {msg}')
    time.sleep(0.2)

    while True:
        raw_data = ser.readline() # 한 줄 읽기
        if len(raw_data) == 0:
            break
        text = raw_data.decode('utf-8', errors='replace').strip()
        if text:
            print(f"            [수신] {text}")
        # time.sleep(1)