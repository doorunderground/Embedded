import serial
import time
PORT = 'COM3' # 직접 확인한 번호 입력   
BAUDRATE = 9600
ser = serial.Serial(PORT, BAUDRATE, timeout=1)
print(f"✅ {PORT} 포트 연결 성공! (속도: {BAUDRATE})")
while True:
    msg = input('>') + '\n'
    msg_bytes = msg.encode()
    ser.write(msg_bytes)
    raw_data = ser.readline() # 한 줄 읽기
    text = raw_data.decode('utf-8', errors='replace').strip()
    if text:
        print(f"[수신] {text}")
        time.sleep(0.01)
