#!/usr/bin/env python
# coding: utf-8

# 
# # YOLOv8 + ByteTrack (Notebook-friendly)
# 
# - **No `argparse`** — runs cleanly inside Jupyter.
# - Plug in your **weights** and **video** paths and run.
# - Saves `vis.mp4` (overlay video) and `tracks.json` in the current working directory.
# 

# In[1]:


# If needed, uncomment and run once:
# %pip install --quiet ultralytics opencv-python scipy


# In[2]:


from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import json, os
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics not found. Please install with `%pip install ultralytics`.") from e

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise RuntimeError("scipy not found. Please install with `%pip install scipy`.") from e

# --- Your 10 class names (order matters) ---
OVERRIDE_NAMES = [
    'bottle','bulletproof_plate','canteen','magazine','gas_mask',
    'boots_black','boots_brown','MRE','gas_mask_pouch','helmet'
]


# In[3]:


BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2

@dataclass
class Detection:
    bbox: BBox
    score: float
    cls: int

@dataclass
class Track:
    track_id: int
    bbox: BBox
    score: float
    cls: int
    age: int = 0
    time_since_update: int = 0

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N,4), b: (M,4) in xyxy
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)


# In[4]:


class YoloV8Detector:
    def __init__(
        self,
        weights: str,
        class_filter: Optional[List[int]] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        override_names: Optional[List[str]] = None,   # optional: force class names
    ):
        self.model = YOLO(weights)
        names_dict = getattr(self.model, "names", None) or getattr(self.model.model, "names")
        if override_names is not None:
            names_dict = {i: n for i, n in enumerate(override_names)}
            # set back to model for consistent visuals/logs
            if hasattr(self.model, "names"):
                self.model.names = names_dict
            else:
                self.model.model.names = names_dict
        self.names: Dict[int, str] = names_dict

        self.class_filter = set(class_filter) if class_filter else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def __call__(self, frame) -> List[Detection]:
        r = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy.numpy()
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf.numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls.numpy().astype(int)
        dets: List[Detection] = []
        for (x1,y1,x2,y2), s, c in zip(xyxy, confs, clss):
            if self.class_filter is None or c in self.class_filter:
                dets.append(Detection((float(x1),float(y1),float(x2),float(y2)), float(s), int(c)))
        return dets


# In[5]:


class SimpleByteTracker:
    def __init__(self,
                 conf_high: float = 0.5,
                 conf_low: float = 0.1,
                 iou_thresh: float = 0.3,
                 max_age: int = 30,
                 min_hits: int = 2,
                 per_class: bool = True):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.per_class = per_class
        self.tracks: List[Track] = []
        self.next_id = 1

    def _match(self, tracks_idxs: List[int], dets: List[Detection]):
        if not tracks_idxs or not dets:
            return [], tracks_idxs, list(range(len(dets)))
        A = np.array([self.tracks[i].bbox for i in tracks_idxs], float)
        B = np.array([d.bbox for d in dets], float)
        iou = iou_matrix(A, B)
        cost = 1.0 - iou
        r, c = linear_sum_assignment(cost)
        matches, u_t, u_d = [], set(range(len(tracks_idxs))), set(range(len(dets)))
        for ri, ci in zip(r, c):
            if 1 - cost[ri, ci] >= self.iou_thresh:
                tr_i = tracks_idxs[ri]
                # class consistency
                if self.tracks[tr_i].cls == dets[ci].cls:
                    matches.append((tr_i, ci))
                    u_t.discard(ri); u_d.discard(ci)
        return matches, [tracks_idxs[i] for i in u_t], list(u_d)

    def update(self, detections: List[Detection]) -> List[Track]:
        # age all
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        high = [d for d in detections if d.score >= self.conf_high]
        low  = [d for d in detections if self.conf_low <= d.score < self.conf_high]

        unmatched_tracks = set(range(len(self.tracks)))

        def group(ds):
            mp = {}
            for d in ds: mp.setdefault(d.cls, []).append(d)
            return mp

        if self.per_class:
            high_map = group(high); low_map = group(low)
            classes = set(list(high_map.keys()) + list(low_map.keys()))
            for cls_id in classes:
                # high first
                ti = [i for i in unmatched_tracks if self.tracks[i].cls == cls_id]
                m, u_t, u_d = self._match(ti, high_map.get(cls_id, []))
                for tr_i, di in m:
                    d = high_map[cls_id][di]
                    self.tracks[tr_i].bbox = d.bbox
                    self.tracks[tr_i].score = d.score
                    self.tracks[tr_i].time_since_update = 0
                    unmatched_tracks.discard(tr_i)
                # then low
                ti = [i for i in unmatched_tracks if self.tracks[i].cls == cls_id]
                m, u_t, u_d = self._match(ti, low_map.get(cls_id, []))
                for tr_i, di in m:
                    d = low_map[cls_id][di]
                    self.tracks[tr_i].bbox = d.bbox
                    self.tracks[tr_i].score = d.score
                    self.tracks[tr_i].time_since_update = 0
                    unmatched_tracks.discard(tr_i)

            # new tracks for unmatched detections (both high and low)
            # simple policy: create new for all remaining detections
            for d in high + low:
                self.tracks.append(Track(self.next_id, d.bbox, d.score, d.cls, age=1, time_since_update=0))
                self.next_id += 1
        else:
            # simpler global matching (not recommended for multi-class)
            ti = list(range(len(self.tracks)))
            m, u_t, u_d = self._match(ti, high)
            used_high = set(di for _, di in m)
            for tr_i, di in m:
                d = high[di]
                self.tracks[tr_i].bbox = d.bbox
                self.tracks[tr_i].score = d.score
                self.tracks[tr_i].time_since_update = 0

            rem_tracks = [i for i in range(len(self.tracks)) if self.tracks[i].time_since_update > 0]
            rem_low = [d for j, d in enumerate(low)]
            m2, u_t2, u_d2 = self._match(rem_tracks, rem_low)
            for tr_i, di in m2:
                d = rem_low[di]
                self.tracks[tr_i].bbox = d.bbox
                self.tracks[tr_i].score = d.score
                self.tracks[tr_i].time_since_update = 0

            for j, d in enumerate(high):
                if j not in used_high:
                    self.tracks.append(Track(self.next_id, d.bbox, d.score, d.cls, age=1, time_since_update=0))
                    self.next_id += 1
            for d in low:
                self.tracks.append(Track(self.next_id, d.bbox, d.score, d.cls, age=1, time_since_update=0))
                self.next_id += 1

        # drop stale
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # output
        return [t for t in self.tracks if (t.age >= self.min_hits or t.time_since_update == 0)]


# In[6]:


def draw_tracks(frame, tracks: List[Track], class_names: Dict[int, str]):
    for t in tracks:
        x1,y1,x2,y2 = map(int, t.bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        name = class_names.get(t.cls, str(t.cls))
        label = f"id{t.track_id}:{name}:{t.score:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame


# In[7]:


# === USER PARAMETERS (edit here) ===

weights_path = r"C:\Users\User\ai\HJ_Project\ai_club\military_classification\best.pt"
video_path   = r"C:\Users\User\ai\HJ_Project\ai_club\military_classification\integrated_video.mp4"

# Track specific classes only (by name). None = track all 10
# Example: class_filter_names = ['bottle','helmet']
class_filter_names = None

# Detector thresholds
DET_CONF = 0.25
DET_IOU  = 0.7
IMG_SIZE = 1280

# Tracker thresholds
CONF_HIGH = 0.5
CONF_LOW  = 0.1
IOU_THRESH = 0.3
MAX_AGE   = 30
MIN_HITS  = 2
PER_CLASS = True

OUT_JSON = "tracks.json"
OUT_MP4  = "vis.mp4"


# In[9]:


class YoloV8Detector:
    def __init__(
        self,
        weights: str,
        class_filter: Optional[List[int]] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        override_names: Optional[List[str]] = None,   # optional: external names map
    ):
        self.model = YOLO(weights)

        # 1) 모델에서 names 읽기 (list or dict 모두 지원) — 읽기 전용이므로 '할당' 금지
        #    우선순위: model.model.names -> model.names
        model_names = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            model_names = self.model.model.names
        elif hasattr(self.model, "names"):
            model_names = self.model.names

        # 2) dict 형태로 통일
        def to_id_name_map(x):
            if x is None:
                return {}
            if isinstance(x, dict):
                return {int(k): str(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return {i: str(n) for i, n in enumerate(x)}
            return {}

        base_map = to_id_name_map(model_names)

        # 3) override가 있으면 '우리 래퍼 내부에서만' 사용 (모델 내부에 set하지 않음)
        if override_names is not None:
            self.names: Dict[int, str] = {i: str(n) for i, n in enumerate(override_names)}
        else:
            self.names: Dict[int, str] = base_map

        # 필터/파라미터
        self.class_filter = set(class_filter) if class_filter else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def __call__(self, frame) -> List[Detection]:
        r = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        # torch.Tensor 또는 numpy 대응
        xyxy = r.boxes.xyxy
        confs = r.boxes.conf
        clss = r.boxes.cls

        if hasattr(xyxy, "cpu"):  # torch
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy().astype(int)
        else:  # 이미 numpy
            xyxy = xyxy.numpy()
            confs = confs.numpy()
            clss = clss.numpy().astype(int)

        dets: List[Detection] = []
        for (x1, y1, x2, y2), s, c in zip(xyxy, confs, clss):
            if self.class_filter is None or c in self.class_filter:
                dets.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(s), int(c)))
        return dets



# In[15]:


# === Notebook version of run_tracker.py (argparse 제거) ===
import os, json, cv2
from IPython.display import Video, display

# ---- 필요 모듈 임포트 (프로젝트 구조에 맞게 조정) ----
# 노트북에 이미 클래스 정의 셀이 있다면 아래 두 줄은 주석 처리해도 됩니다.
#from src.detector_yolo import YoloV8Detector   # ← 이전에 수정한 "names setter 안 건드리는" 버전이어야 함
#from src.tracker_byte import SimpleByteTracker
#from src.vis import draw_tracks

# ---- 유틸: '0,1,2' 또는 'bottle,helmet' 모두 지원 ----
def parse_class_ids(arg, names_map: dict[int, str] | None):
    if not arg:
        return None
    if isinstance(arg, (list, tuple)):   # ['bottle','helmet'] 처럼 리스트로 받은 경우
        tokens = [str(t).strip() for t in arg if str(t).strip()]
    else:                                 # "bottle,helmet" 또는 "0,9" 문자열
        tokens = [t.strip() for t in str(arg).split(",") if t.strip()]
    id_list = []
    inv = {v: k for k, v in names_map.items()} if names_map else {}
    for tok in tokens:
        if tok.isdigit():
            id_list.append(int(tok))
        else:
            if tok not in inv:
                raise ValueError(f"Unknown class name: {tok}. Known: {list(inv.keys())}")
            id_list.append(inv[tok])
    return id_list

# ===========================================
# ============== 사용자 설정 =================
# ===========================================
weights_path = r"C:\Users\User\ai\HJ_Project\ai_club\military_classification\best.pt"
video_path   = r"C:\Users\User\ai\HJ_Project\ai_club\military_classification\integrated_video.mp4"

# 가중치 내부 names가 다르거나 비어있으면 강제 덮어쓰기(모델 내부에 set 하지 말고, 래퍼 내부에서만 사용)
OVERRIDE_NAMES = ["magazine", "bottle", "boots_black", "boots_brown", "bulletproof_plate",
               "canteen", "gas_mask", "gas_mask_pouch", "helmet", "MRE"]

# 추적할 클래스 제한 (없으면 None) — 이름 리스트 또는 "이름,이름" 또는 "0,9" 모두 허용
classes_to_track = None  # 예: ['bottle','helmet'] 또는 "bottle,helmet" 또는 "0,9"

# Detector 설정
DET_CONF = 0.25
DET_IOU  = 0.7
IMG_SIZE = 1280

# Tracker 설정(ByteTrack 스타일)
CONF_HIGH = 0.5
CONF_LOW  = 0.1
IOU_THRESH = 0.3
MAX_AGE   = 30
MIN_HITS  = 2
PER_CLASS = True

# 출력 파일
OUT_JSON = "tracks.json"
OUT_MP4  = "vis.mp4"
SHOW_IN_NOTEBOOK = True     # 결과 영상을 노트북에서 바로 표시

# ===========================================
# ============== 실행 함수 ==================
# ===========================================
def run_notebook(video_path: str,
                 weights_path: str,
                 classes_to_track=None,
                 override_names=None,
                 out_json="tracks.json",
                 out_mp4="vis.mp4"):
    # Detector
    det = YoloV8Detector(
        weights=weights_path,
        class_filter=None,           # 아래에서 name->id 변환 후 세팅
        conf=DET_CONF, iou=DET_IOU, imgsz=IMG_SIZE,
        override_names=override_names
    )
    print("Class map:", det.names)

    # 클래스 필터 적용
    class_filter = parse_class_ids(classes_to_track, det.names) if classes_to_track else None
    if class_filter is not None:
        det.class_filter = set(class_filter)

    # Tracker
    trk = SimpleByteTracker(
        conf_high=CONF_HIGH, conf_low=CONF_LOW,
        iou_thresh=IOU_THRESH, max_age=MAX_AGE, min_hits=MIN_HITS,
        per_class=PER_CLASS
    )

    # 비디오 IO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    if not writer.isOpened():
        # 환경에 따라 mp4 코덱이 안 먹힐 수 있어요. 임시로 XVID/AVI로 폴백합니다.
        print("[WARN] mp4v writer open failed. Falling back to XVID/AVI.")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_mp4 = os.path.splitext(out_mp4)[0] + ".avi"
        writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
        assert writer.isOpened(), "VideoWriter failed to open. Check codecs/paths."

    all_out = []
    frames_read = 0
    frames_written = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1

            dets = det(frame)
            tracks = trk.update(dets)

            vis = draw_tracks(frame, tracks, det.names)
            # 크기/타입 보정
            if vis.shape[1] != w or vis.shape[0] != h:
                vis = cv2.resize(vis, (w, h))
            if vis.dtype != "uint8":
                vis = vis.astype("uint8")

            writer.write(vis)
            frames_written += 1

            all_out.append({
                "frame": frames_read-1,
                "tracks": [
                    {"id": t.track_id, "bbox": list(map(float, t.bbox)),
                     "score": float(t.score), "cls": int(t.cls)} for t in tracks
                ]
            })
    finally:
        cap.release()
        writer.release()

    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(all_out, fp, ensure_ascii=False)

    print(f"frames_read={frames_read}, frames_written={frames_written}")
    print("Saved:", os.path.abspath(out_json), os.path.abspath(out_mp4))

    if SHOW_IN_NOTEBOOK and frames_written > 0:
        try:
            display(Video(filename=out_mp4, embed=True))
        except Exception as e:
            print("[INFO] Inline display failed, but the video file was saved.", e)

# ===========================================
# ============== 실행 =======================
# ===========================================
run_notebook(
    video_path=video_path,
    weights_path=weights_path,
    classes_to_track=classes_to_track,
    override_names=OVERRIDE_NAMES
)


# In[10]:


from IPython.display import Video, display

# 결과 파일 경로 (필요하면 바꿔도 됨)
result_video = "vis.mp4"  # 또는 r"C:\경로\vis.mp4"

display(Video(filename=result_video, embed=True))


# 확인

# In[ ]:





# In[ ]:




