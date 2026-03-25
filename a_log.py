import csv
from collections import Counter

file_path = 'snack_test/zec/snack_test_log.csv'

def analyze_snack_logs(path):
    best_detections = {}
    fail_count = 0

    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3: continue
                
                timestamp = row[0].strip()
                class_id = row[1].strip()
                try:
                    confidence = float(row[2].strip())
                except ValueError:
                    continue

                if confidence <= 0.0:
                    if timestamp not in best_detections:
                        best_detections[timestamp] = ("FAIL", 0.0)
                    continue

                if timestamp not in best_detections or \
                   best_detections[timestamp][0] == "FAIL" or \
                   confidence > best_detections[timestamp][1]:
                    best_detections[timestamp] = (class_id, confidence)

        results = [val[0] for val in best_detections.values()]
        final_counts = Counter(results)

        print("\n" + "="*35)
        print(f"      과자 테스트 최종 집계")
        print("="*35)
        print(f"전체 시도(프레임): {len(best_detections)}개")
        print("-" * 35)
        
        for i in range(10):
            cid = str(i)
            count = final_counts.get(cid, 0)
            print(f" 클래스 {cid:2} : {count:3} 회")
            
        print("-" * 35)
        print(f" 검출 실패  : {final_counts.get('FAIL', 0):3} 회")
        print("="*35)

    except FileNotFoundError:
        print(f"에러: '{path}' 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    analyze_snack_logs(file_path)