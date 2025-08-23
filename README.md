# Military Supply Detection (YOLOv8/YOLO11 기반)

군수품 객체 인식을 위한 YOLO 기반 파이프라인입니다.  
**영상 → 프레임 추출 → 일부 라벨링 → 자동 라벨 보정 → 데이터셋 생성 → 학습 → 추적/카운팅**  
흐름으로 구성되어 있습니다.

---

## 🚀 주요 스크립트 & 실행 순서

1. **프레임 추출 (`frames.py`)**
   - 영상에서 균등 간격으로 프레임을 추출하여 `frames/<class>` 폴더에 저장합니다.
   ```bash
   python scripts/frames.py
   ```

2. **수작업 라벨링**
   - 추출된 프레임 중 일부를 [LabelImg](https://github.com/heartexlabs/labelImg) 등 도구로 직접 라벨링합니다.
   - 라벨은 `labels/<class>` 폴더에 저장합니다.

3. **자동 라벨링 (`autolabelling.py`)**
   - 수작업 라벨을 이용해 소규모 학습 후, 나머지 이미지에 대해 자동 라벨링을 수행합니다.
   - 기존 라벨은 덮어쓰지 않으며, 라벨 없는 이미지에만 결과를 생성합니다.
   ```bash
   python scripts/autolabelling.py
   ```

4. **데이터셋 구성 (`create_dataset.py`)**
   - `images/`, `labels/` 폴더를 읽어 YOLO 학습 가능한 구조로 재구성합니다.
   - 결과: `dataset/images/train|val`, `dataset/labels/train|val`, `dataset/data.yaml`
   ```bash
   python scripts/create_dataset.py
   ```

5. **YOLO 학습**
   - 생성된 `data.yaml`을 이용해 YOLO 모델을 학습합니다.
   ```bash
   yolo detect train model=yolo11n.pt data=dataset/data.yaml imgsz=640 epochs=200 batch=16
   ```

6. **단일 이미지 예측 (`test.py`)**
   - 학습된 모델을 불러와 단일 이미지에 대해 예측을 수행합니다.
   ```bash
   python scripts/test.py
   ```

7. **영상 추적 + 카운팅 (`track_count.py`)**
   - YOLO + ByteTrack 기반으로 객체를 추적하며, 특정 라인을 기준으로 교차 카운팅을 수행합니다.
   ```bash
   python scripts/track_count.py
   ```
