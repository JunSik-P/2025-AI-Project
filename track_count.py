import cv2
from ultralytics import YOLO
from collections import defaultdict

# 1. 모델 불러오기
model = YOLO(r"C:\Users\User\Desktop\jo\PROJECT\runs\detect\train5\weights\best.pt")

# 클래스 이름 매핑
class_names = model.names

# 2. 카운트 저장용 딕셔너리
counts = defaultdict(int)   # 클래스 이름별 카운트
track_history = {}          # id별 이전 중심 좌표 저장

# 3. 비디오 읽기
video_path = r"C:\Users\User\Desktop\jo\PROJECT\test_video\output.mp4"
cap = cv2.VideoCapture(video_path)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out_path = r"C:\Users\User\Desktop\jo\PROJECT\result_counted.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ✅ [사용자 설정] 선의 두 점
LINE_START = (0, 0)   # (x1, y1)
LINE_END   = (720, 0)   # (x2, y2)

# ========================
# 교차 판정 함수 (두 선분 교차 여부)
# ========================
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# 4. 프레임 반복
for result in model.track(source=video_path, tracker="bytetrack.yaml", stream=True):
    frame = result.orig_img.copy()

    if result.boxes.id is None:
        writer.write(frame)
        continue

    # 탐지된 박스들
    for box, track_id, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.id.int().cpu().numpy(),
                                  result.boxes.cls.int().cpu().numpy()):
        x1, y1, x2, y2 = box
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        # id 별 이전 중심 좌표 가져오기
        prev_point = track_history.get(track_id, (cx, cy))
        track_history[track_id] = (cx, cy)

        # 클래스 이름 변환
        cls_name = class_names[int(cls)]

        # 선 교차 여부 확인
        if intersect(prev_point, (cx, cy), LINE_START, LINE_END):
            counts[cls_name] += 1

        # 박스 & id + 클래스 표시
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"id{track_id} {cls_name}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 선 그리기
    cv2.line(frame, LINE_START, LINE_END, (0,0,255), 2)

    # 카운트 출력 (좌상단)
    y_offset = 30
    for cls_name, cnt in counts.items():
        cv2.putText(frame, f"{cls_name}: {cnt}", (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        y_offset += 30

    writer.write(frame)

cap.release()
writer.release()
print("✅ 결과 저장 완료:", out_path)
