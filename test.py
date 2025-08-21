from ultralytics import YOLO

# 1. 모델 로드 (학습된 best.pt 경로 지정)
model = YOLO(r"runs/detect/train5/weights/best.pt")

# 2. 예측할 이미지 경로
source = r"image.png"

# 3. 예측 수행 (결과를 result/predict에 저장)
results = model.predict(
    source=source,
    save=True,           # 결과 이미지 저장
    imgsz=640,           # 32의 배수 (720 → 736)
    conf=0.01,           # confidence threshold (낮춰서 더 탐지 유도)
    iou=0.55,            # NMS IoU threshold
    line_width=2,        # 바운딩 박스 선 두께
    project="result",    # 저장될 상위 폴더
    name="predict",      # 하위 폴더 이름 (result/predict/)
    exist_ok=True        # 이미 있어도 덮어쓰기 허용
)

# 4. 결과 경로 출력
print("✅ 예측 결과 저장 경로:", results[0].save_dir)
