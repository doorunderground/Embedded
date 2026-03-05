# STM32 의 역할 , SLAVE로 동작
# ECHO 서버

import serial
import time

PORT = 'COM4' #CH340의 포트
BAUDRATE = 9600

ser = serial.Serial(PORT, BAUDRATE, timeout=1)
print(f"✅ {PORT} 포트 연결 성공! (속도: {BAUDRATE})")
while True:
    # 읽기를 먼저
    try:
        raw_data = ser.readline() # 한 줄 읽기
        text = raw_data.decode('utf-8', errors='replace').strip()
        print(f"--- CH340 수신 : {text}")
        
        msg_bytes = (text + '\n').encode()
        ser.write(msg_bytes)
    except:
        print("---")