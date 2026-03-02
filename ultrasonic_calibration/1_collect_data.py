"""
1_collect_data.py
─────────────────
초음파 측정값과 자로 잰 실측값을 data.csv에 추가합니다.
실행: python 1_collect_data.py
"""

import csv
import os

DATA_FILE = "data.csv"

def load_existing():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["measured_cm", "actual_cm"])
        return []

    with open(DATA_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

def save_row(measured, actual):
    with open(DATA_FILE, "a", newline="") as f:
        csv.writer(f).writerow([measured, actual])

def main():
    print("=" * 45)
    print("  초음파 센서 보정 데이터 수집기")
    print("=" * 45)

    rows = load_existing()
    print(f"\n현재 저장된 데이터: {len(rows)}개")
    if rows:
        print(f"{'측정(cm)':>10}  {'실측(cm)':>10}")
        print("-" * 25)
        for r in rows:
            print(f"{float(r['measured_cm']):>10.1f}  {float(r['actual_cm']):>10.1f}")

    print("\n새 데이터를 입력하세요. (종료: q)")
    print("─" * 45)

    while True:
        measured_input = input("\n초음파 측정값 (cm): ").strip()
        if measured_input.lower() == "q":
            break
        actual_input = input("자로 잰 실측값 (cm): ").strip()
        if actual_input.lower() == "q":
            break

        try:
            measured = float(measured_input)
            actual   = float(actual_input)
        except ValueError:
            print("  숫자를 입력하세요.")
            continue

        save_row(measured, actual)
        print(f"  저장 완료: 측정 {measured}cm → 실측 {actual}cm")

    rows = load_existing()
    print(f"\n총 {len(rows)}개 데이터가 {DATA_FILE}에 저장되어 있습니다.")
    print("다음 단계: python 2_train_models.py")

if __name__ == "__main__":
    main()