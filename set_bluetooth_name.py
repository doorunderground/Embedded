import serial
import time

# 설정값
PORT = 'COM4' 
BAUDRATE = 9600
NEW_NAME = "UNDERGROUND" # 변경하고 싶은 이름 입력 (영문/숫자 권장)

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트 연결 성공! (속도: {BAUDRATE})")
    time.sleep(1) # 안정적인 통신을 위한 대기

    # 1. 통신 테스트 (AT)
    print(f"📡 'AT' 명령어 송신 중...")
    ser.write("AT".encode('utf-8'))
    time.sleep(2.0)

    response = ser.read_all().decode('utf-8', errors='replace').strip()
    if "OK" in response:
        print(f"[수신] {response} (통신 정상)")

        # 2. 이름 변경 (AT+NAME)
        cmd = f"AT+NAME{NEW_NAME}"
        print(f"📡 이름 변경 요청: {cmd}")
        ser.write(cmd.encode('utf-8'))
        time.sleep(2.0)  # 쓰기 작업 완료를 위한 충분한 대기

        name_response = ser.read_all().decode('utf-8', errors='replace').strip()
        print(f"[수신] {name_response}")

        if "OKsetname" in name_response or "OK" in name_response:
            print(f"✨ 성공: 이름이 '{NEW_NAME}'으로 변경되었습니다.")
        else:
            print("⚠️ 응답이 올바르지 않습니다. 다시 시도하십시오.")

    else:
        print("❌ 'OK' 응답이 없습니다. 배선 또는 보드레이트(9600/38400)를 확인하세요.")

except Exception as e:
    print(f"❌ 오류 발생: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("🔌 포트 연결 종료")

