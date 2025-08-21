# pip install ultralytics supervision opencv-python

import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker import BYTETracker, BYTETrackerArgs
from supervision.tracker.byte_tracker.utils import detections2boxes, match_detections_with_tracks
from tqdm.auto import tqdm

# ====== 사용자 설정 ======
SOURCE_VIDEO1_PATH = r"C:\path\to\input.mp4"   # ← 입력 동영상 경로
TARGET_VIDEO1_PATH = r"C:\path\to\result.mp4"  # ← 출력 동영상 경로
MODEL_WEIGHTS = "yolo11s.pt"                   # or 학습한 best.pt

# 선(카운팅 라인) 위치
LINE_START = sv.Point(540, 50)
LINE_END   = sv.Point(540, 720 - 50)

# 특정 클래스만 추적/표시하고 싶다면 집합으로 지정, 아니면 None
# 예: CLASS_FILTER = {0, 1, 5}
CLASS_FILTER = None

# ====== 모델/도구 준비 ======
model = YOLO(MODEL_WEIGHTS)

# 클래스 이름 매핑
try:
    names = model.model.names
except AttributeError:
    names = model.names
if isinstance(names, dict):
    CLASS_NAMES_DICT = {int(k): v for k, v in names.items()}
else:
    CLASS_NAMES_DICT = {i: n for i, n in enumerate(list(names))}

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO1_PATH)
byte_tracker = BYTETracker(BYTETrackerArgs())

# 프레임 제너레이터
generator = sv.get_video_frames_generator(SOURCE_VIDEO1_PATH)

# 라인 카운터 및 어노테이터
line_counter = sv.LineCounter(start=LINE_START, end=LINE_END)
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.5
)
line_annotator = sv.LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

# ====== 메인 루프 ======
with sv.VideoSink(TARGET_VIDEO1_PATH, video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
        # YOLO 추론
        results = model(frame, verbose=False)[0]

        # Ultralytics -> supervision.Detections 변환
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

        # 클래스 필터링(옵션)
        if CLASS_FILTER is not None:
            mask = np.isin(detections.class_id, list(CLASS_FILTER))
            detections.filter(mask=mask, inplace=True)

        # ByteTrack 업데이트
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )

        # detection ↔ track 매칭하여 tracker_id 부여
        tracker_ids = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_ids)

        # 트래커 없는 detection 제거
        mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # 라벨 텍스트 구성
        labels = [
            f"#{tid} {CLASS_NAMES_DICT.get(cid, cid)} {conf:0.2f}"
            for cid, conf, tid in zip(detections.class_id, detections.confidence, detections.tracker_id)
        ]

        # 라인 카운트 업데이트
        line_counter.update(detections=detections)

        # 시각화
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        # 프레임 저장
        sink.write_frame(frame)

print(f"[✓] Saved → {TARGET_VIDEO1_PATH}")
print(f"[i] Line counts | In: {line_counter.in_count}  Out: {line_counter.out_count}")
