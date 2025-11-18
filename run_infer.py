# -*- coding: utf-8 -*-
"""
Unified Inference (YOLO items + InsightFace faces) — v3.3
- 트래커를 루프 밖에서 1회 생성 (트랙 ID 안정화)
- NO_FACE, NO_ITEM 에러를 ui_cues.jsonl로 출력 (각 2.0s 지속 시 확정)
- event_id 부여, summary.outputs 경로 유지 (GUI 연동)
- FACE_BANK_DIR 변수 사용 수정
"""

import os, sys, json, re, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Deque
from collections import defaultdict, Counter, deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO as UL_YOLO
from insightface.app import FaceAnalysis

# ======================== 기본 경로 =========================
ROOT = Path(__file__).resolve().parent
DEF_YOLO_WEIGHT = ROOT / "models" / "yolo" / "detector.pt"
DEF_CLASSES_TXT = ROOT / "config" / "classes.txt"
DEF_FACE_BANK_DIR = ROOT / "face_bank"
DEF_BASE_OUTPUT_ROOT = ROOT / "output"

# ========================== 유틸리티 ============================
def cosine_sim(a, b, eps=1e-9):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps)))

def draw_box(img, box, color, text=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if text:
        tw = max(80, 10 + 9*len(text))
        y0 = max(0, y1-22)
        cv2.rectangle(img, (x1, y0), (x1+tw, y1), color, -1)
        cv2.putText(img, text, (x1+5, max(y0+16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def ensure_writer(out_path, w, h, fps):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), (w, h))

def load_classes(classes_txt: Optional[Path]):
    if not classes_txt or not classes_txt.exists():
        return None
    names = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names if names else None

def sanitize_folder_name(name: str) -> str:
    name = name.strip()
    if not name: return ""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name[:100]

def next_nonconflicting_dir(base_dir: Path) -> Path:
    if not base_dir.exists(): return base_dir
    i = 1
    while True:
        cand = base_dir.parent / f"{base_dir.name}_{i}"
        if not cand.exists(): return cand
        i += 1

# ===================== 트래커/클래스 잠금 =====================
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: int
    hits: int
    miss: int
    score: float

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    return float(inter / max(1e-6, (area_a + area_b - inter)))

class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.25, max_miss: int = 30):
        self.iou_thresh = iou_thresh; self.max_miss = max_miss
        self.next_id = 1; self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[Tuple[int,int,int,int, float]], frame_idx: int) -> Dict[int, Track]:
        det_bboxes = [d[:4] for d in dets]; det_scores = [d[4] for d in dets]
        tids = list(self.tracks.keys()); tbxs = [self.tracks[t].bbox for t in tids]
        iou_mat = np.zeros((len(tbxs), len(det_bboxes)), dtype=np.float32)
        for i_, tb in enumerate(tbxs):
            for j_, db in enumerate(det_bboxes): iou_mat[i_, j_] = iou(tb, db)
        matched_t, matched_d = set(), set(); pairs = []
        for _ in range(min(len(tbxs), len(det_bboxes))):
            i_, j_ = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[i_, j_] < self.iou_thresh: break
            if i_ in matched_t or j_ in matched_d: iou_mat[i_, j_] = -1; continue
            matched_t.add(i_); matched_d.add(j_); pairs.append((tids[i_], j_))
            iou_mat[i_, :] = -1; iou_mat[:, j_] = -1
        for tid, j_ in pairs:
            self.tracks[tid].bbox = tuple(map(int, det_bboxes[j_])); self.tracks[tid].last_seen = frame_idx
            self.tracks[tid].hits += 1; self.tracks[tid].miss = 0; self.tracks[tid].score = float(det_scores[j_])
        for j_, db in enumerate(det_bboxes):
            if j_ not in matched_d:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(tid, tuple(map(int, db)), frame_idx, 1, 0, float(det_scores[j_]))
        to_del = []
        for tid in list(self.tracks.keys()):
            if tid not in [t for t, _ in pairs]: self.tracks[tid].miss += 1
            if self.tracks[tid].miss > self.max_miss: to_del.append(tid)
        for tid in to_del: del self.tracks[tid]
        return self.tracks

# ==== 게이트/하이퍼파라 ====
IOU_MIN = 0.08
UPPER_RATIO = 0.85
CLASS_STABLE_WINDOW_S = 0.8
AGREEMENT_RATIO_MIN = 0.7
COOLDOWN_S = 2.0
NO_ITEM_WINDOW_S = 1.0

CLASS_CONF_THR: Dict[str, float] = {
    "magazine": 0.45, "bottle": 0.45, "boots_black": 0.50, "boots_brown": 0.50,
    "bulletproof_plate": 0.40, "canteen": 0.45, "gas_mask": 0.45,
    "gas_mask_pouch": 0.45, "helmet": 0.45, "MRE": 0.45,
}
DEFAULT_CONF_THR = 0.45

@dataclass
class ClassLockState:
    state: str = "IDLE"
    clazz: Optional[str] = None
    locked_at_s: float = 0.0
    cooldown_until_s: float = 0.0
    last_seen_s: float = 0.0

from collections import defaultdict as _dd
class StabilityBuffer:
    def __init__(self):
        self.buf: Dict[int, Deque[Tuple[float, str, float]]] = _dd(lambda: deque(maxlen=120))
    def push(self, item_tid: int, ts: float, clazz: str, conf: float):
        self.buf[item_tid].append((ts, clazz, conf))
    def agreement(self, item_tid: int, ts: float, target_class: str, window_s: float) -> float:
        dq = self.buf[item_tid]; start = ts - window_s
        recent = [c for (t, c, _conf) in dq if t >= start]
        if not recent: return 0.0
        cnt = sum(1 for c in recent if c == target_class)
        return cnt / max(1, len(recent))
    def recent_conf(self, item_tid: int, ts: float, window_s: float, target_class: str) -> float:
        dq = self.buf[item_tid]; start = ts - window_s
        vals = [_conf for (t, c, _conf) in dq if t >= start and c == target_class]
        return float(np.median(vals)) if vals else 0.0

class ClassLockGate:
    def __init__(self):
        self.person_state: Dict[int, ClassLockState] = _dd(ClassLockState)
        self.stab = StabilityBuffer()
    def update_buffers(self, item_track_id: int, ts: float, clazz: str, conf: float):
        self.stab.push(item_track_id, ts, clazz, conf)
    def decide(self, pid: int, item_tid: int, ts: float, clazz: str, conf: float, cond_upper: bool, cond_iou: bool):
        ps = self.person_state[pid]; ps.last_seen_s = ts
        if not (cond_upper and cond_iou): return False, "gate_cond_not_met"
        agree = self.stab.agreement(item_tid, ts, clazz, CLASS_STABLE_WINDOW_S)
        conf_med = self.stab.recent_conf(item_tid, ts, CLASS_STABLE_WINDOW_S, clazz)
        conf_thr = CLASS_CONF_THR.get(clazz, DEFAULT_CONF_THR)
        if agree < AGREEMENT_RATIO_MIN or conf_med < conf_thr:
            return False, "stability_not_met"
        if ps.state == "IDLE":
            ps.state = "HOLDING"; ps.clazz = clazz; ps.locked_at_s = ts; ps.cooldown_until_s = ts + COOLDOWN_S
            return True, "locked_and_allowed"
        else:
            if ps.clazz != clazz: return False, "class_locked"
            return True, "holding_allowed"
    def release_if_needed(self, pid: int, ts: float, has_any_item: bool):
        ps = self.person_state[pid]
        if ps.state == "HOLDING":
            if not has_any_item and (ts - ps.last_seen_s) >= NO_ITEM_WINDOW_S:
                ps.state = "IDLE"; ps.clazz = None

def box_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def is_item_in_upper_body(person_box, item_box, upper_ratio: float = UPPER_RATIO) -> bool:
    px1,py1,px2,py2 = person_box
    cx, cy = box_center(item_box)
    upper_y = int(py1 + (py2 - py1) * upper_ratio)
    return (px1 <= cx <= px2) and (py1 <= cy <= upper_y)

# =========================== 에러 게이트 ===============================
class DurGate:
    """상태가 지속될 때만 True를 1회 출력"""
    def __init__(self, fps: float, hold_sec: float):
        self.req_frames = max(1, int(round(fps * hold_sec)))
        self.counter = 0
        self.active = False
    def evaluate(self, condition: bool):
        if condition:
            self.counter += 1
            if not self.active and self.counter >= self.req_frames:
                self.active = True
                return True
            return False
        else:
            self.counter = 0
            self.active = False
            return False

# =========================== 메인 ===============================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="입력 영상 경로")
    ap.add_argument("--out-root", default=str(DEF_BASE_OUTPUT_ROOT), help="출력 루트 폴더")
    ap.add_argument("--out-name", default=None, help="출력 하위 폴더명 (기본: run_YYYYmmdd_HHMMSS)")
    ap.add_argument("--yolo-weight", default=str(DEF_YOLO_WEIGHT))
    ap.add_argument("--classes", default=str(DEF_CLASSES_TXT))
    ap.add_argument("--face-bank", default=str(DEF_FACE_BANK_DIR))
    ap.add_argument("--yolo-conf", type=float, default=0.30)
    ap.add_argument("--yolo-iou", type=float, default=0.50)
    ap.add_argument("--face-thr", type=float, default=0.40)
    return ap.parse_args()

def main():
    args = parse_args()

    YOLO_WEIGHT = Path(args.yolo_weight)
    CLASSES_TXT = Path(args.classes) if args.classes else None
    FACE_BANK_DIR = Path(args.face_bank)
    BASE_OUTPUT_ROOT = Path(args.out_root)

    print("=== Unified Inference — v3.3 (CLI) ===")
    if not YOLO_WEIGHT.exists(): print(f"❌ YOLO 가중치 없음: {YOLO_WEIGHT}"); sys.exit(1)
    class_names = load_classes(CLASSES_TXT if CLASSES_TXT and CLASSES_TXT.exists() else None)
    print(f"ℹ️ classes: {len(class_names) if class_names else 0} loaded")

    sub = sanitize_folder_name(args.out_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    OUTPUT_ROOT = next_nonconflicting_dir(BASE_OUTPUT_ROOT / sub)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"▶ 출력 폴더: {OUTPUT_ROOT}")

    video_path = Path(args.video).resolve()
    if not video_path.exists(): print(f"❌ 입력 영상 없음: {video_path}"); sys.exit(2)
    base = video_path.stem

    out_video    = OUTPUT_ROOT / f"{base}_result.mp4"
    out_jsonl    = OUTPUT_ROOT / f"{base}_result.jsonl"
    out_events   = OUTPUT_ROOT / f"{base}_events.jsonl"
    out_snaps    = OUTPUT_ROOT / f"{base}_snaps"
    out_holding  = OUTPUT_ROOT / f"{base}_holding_log.jsonl"  # (옵션) 필요 시 기록
    out_summary  = OUTPUT_ROOT / f"{base}_summary.json"
    out_ui_cues  = OUTPUT_ROOT / f"{base}_ui_cues.jsonl"
    out_gate     = OUTPUT_ROOT / f"{base}_gate_log.jsonl"
    out_snaps.mkdir(parents=True, exist_ok=True)

    yolo_conf = float(args.yolo_conf); yolo_iou = float(args.yolo_iou); face_thr = float(args.face_thr)

    yolo = UL_YOLO(str(YOLO_WEIGHT))
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640,640))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): print("❌ 영상을 열 수 없습니다."); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = ensure_writer(out_video, w, h, fps)
    jf = out_jsonl.open("w", encoding="utf-8")
    event_f = out_events.open("w", encoding="utf-8")
    diag_f = out_holding.open("w", encoding="utf-8")
    ui_f = out_ui_cues.open("w", encoding="utf-8")
    gate_f = out_gate.open("w", encoding="utf-8")

    # 트래커는 루프 밖에서 1회 생성
    person_tracker = SimpleTracker(iou_thresh=0.25, max_miss=int(2*fps))
    item_tracker   = SimpleTracker(iou_thresh=0.25, max_miss=int(2*fps))
    item_label_hist: Dict[int, Counter] = defaultdict(Counter)
    last_conf_by_track: Dict[int, float] = defaultdict(float)
    gate = ClassLockGate()

    # 에러 게이트: 2.0s 지속
    no_face_gate = DurGate(fps=fps, hold_sec=2.0)
    no_item_gate = DurGate(fps=fps, hold_sec=2.0)

    def make_event_id(pid: int, iid: int, ts_ms: int) -> str:
        return f"p{pid}_i{iid}_{ts_ms}"

    def on_hold_event(pid, iid, t_s, duration_s, frame, person_box, item_box,
                      person_name_by_track, item_class_by_track,
                      cond_upper, cond_iou):
        iname = item_class_by_track.get(iid, "unknown")
        conf_med = gate.stab.recent_conf(iid, t_s, CLASS_STABLE_WINDOW_S, iname)
        allowed, reason = gate.decide(pid=pid, item_tid=iid, ts=t_s, clazz=iname, conf=conf_med,
                                      cond_upper=cond_upper, cond_iou=cond_iou)
        gate_rec = {"ts_s": float(t_s), "pid": int(pid), "iid": int(iid),
                    "class": iname, "conf_med": float(conf_med),
                    "reason": reason, "allowed": bool(allowed)}
        gate_f.write(json.dumps(gate_rec, ensure_ascii=False) + "\n"); gate_f.flush()
        if not allowed: return
        x1,y1,x2,y2 = map(int, item_box)
        H, W = frame.shape[:2]
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(x1+1, min(W, x2)); y2 = max(y1+1, min(H, y2))
        crop = frame[y1:y2, x1:x2].copy(); ts_ms = int(t_s*1000)
        snap_name = f"event_p{pid}_i{iid}_{ts_ms}.jpg"
        cv2.imwrite(str(out_snaps / snap_name), crop)
        pname, psim, _ = person_name_by_track.get(pid, ("Unknown", -1.0, 0.0))
        evt = {
            "event_id": make_event_id(pid, iid, ts_ms),
            "timestamp_s": float(t_s), "duration_s": float(duration_s),
            "person_track_id": int(pid), "item_track_id": int(iid),
            "person_box": [int(v) for v in person_box], "item_box": [int(v) for v in item_box],
            "person_name": pname, "person_similarity": float(psim),
            "item_class": iname, "status": "needs_face_front" if str(pname).lower()=="unknown" else "pending",
            "snapshot": snap_name
        }
        event_f.write(json.dumps(evt, ensure_ascii=False) + "\n"); event_f.flush()
        print(f"[EVENT] HOLD≥2.0s @ {t_s:.2f}s  person:{pid}({pname})  item:{iid}({iname})  dur:{duration_s:.2f}s")

    print(f"\n▶ 시작: {video_path.name}  {w}x{h} @ {fps:.1f}fps\n▶ 출력 폴더: {OUTPUT_ROOT}\n")

    frame_idx = 0; total_faces = 0; total_items = 0
    hold_started_at: Dict[Tuple[int,int], float] = {}
    last_seen_at: Dict[Tuple[int,int], float] = {}
    fired: Dict[Tuple[int,int], bool] = defaultdict(lambda: False)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; t_s = frame_idx / fps

            try:
                yres = yolo.predict(source=frame, conf=yolo_conf, iou=yolo_iou, device=0, half=True, verbose=False)[0]
            except Exception:
                yres = yolo.predict(source=frame, conf=yolo_conf, iou=yolo_iou, device=0, half=False, verbose=False)[0]

            item_dets = []; yolo_person_dets = []
            if yres and yres.boxes is not None:
                boxes = yres.boxes.xyxy.cpu().numpy(); scores = yres.boxes.conf.cpu().numpy(); clss = yres.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(boxes, scores, clss):
                    cname = str(c)
                    if class_names is not None and 0 <= c < len(class_names):
                        cname = class_names[c]
                    if cname == "person":
                        yolo_person_dets.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s)))
                        draw_box(frame, b, (0, 200, 200), f"person {s:.2f}")
                    else:
                        item_dets.append({"bbox_xyxy": [float(v) for v in b], "score": float(s), "class_id": int(c), "class_name": cname})
                        draw_box(frame, b, (60,180,255), f"{cname} {s:.2f}")

            face_dets = []
            faces = app.get(frame)
            for f in faces:
                x1,y1,x2,y2 = map(float, f.bbox)
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)
                pid, sim = "Unknown", -1.0
                if emb is not None and FACE_BANK_DIR.exists():
                    best_id, best_sim = "Unknown", -1.0
                    for pdir in sorted([d for d in FACE_BANK_DIR.iterdir() if d.is_dir()]):
                        mean_p = pdir / "mean_embedding.npy"
                        if mean_p.exists():
                            gvec = np.load(str(mean_p)); gvec = gvec.mean(axis=0) if gvec.ndim>1 else gvec
                            s = cosine_sim(emb, gvec.astype(np.float32))
                            if s > best_sim: best_sim, best_id = s, pdir.name
                    if best_sim >= face_thr: pid, sim = best_id, best_sim
                face_dets.append({"bbox_xyxy": [x1,y1,x2,y2], "person_id": pid, "similarity": float(sim)})
                draw_box(frame, (x1,y1,x2,y2), (0,200,0) if pid!="Unknown" else (0,100,255), f"{pid} {sim:.2f}" if pid!="Unknown" else "Unknown")
            total_faces += len(face_dets)

            person_inputs: List[Tuple[int,int,int,int, float]] = []
            if yolo_person_dets:
                person_inputs = yolo_person_dets
            else:
                H, W = frame.shape[:2]
                for fd in face_dets:
                    fx1, fy1, fx2, fy2 = map(int, fd["bbox_xyxy"])
                    fw, fh = fx2 - fx1, fy2 - fy1
                    if fw <= 0 or fh <= 0: continue
                    px1 = int(fx1 - 0.5*fw); px2 = int(fx2 + 0.5*fw)
                    py1 = int(fy1 - 0.5*fh); py2 = int(fy2 + 3.0*fh)
                    px1 = max(0, px1); py1 = max(0, py1); px2 = min(W, px2); py2 = min(H, py2)
                    if px2 - px1 < 2 or py2 - py1 < 2: continue
                    person_inputs.append((px1, py1, px2, py2, 0.80))
                    cv2.rectangle(frame, (px1,py1), (px2,py2), (0,255,255), 1)
                    cv2.putText(frame, "upper-body est.", (px1, max(0, py1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

            item_inputs = []
            for it in item_dets:
                x1,y1,x2,y2 = map(int, it["bbox_xyxy"]); sc = float(it["score"])
                item_inputs.append((x1,y1,x2,y2,sc))

            # 트래커 업데이트 (루프 밖에서 생성됨)
            person_tracks = person_tracker.update(person_inputs, frame_idx)
            item_tracks   = item_tracker.update(item_inputs, frame_idx)
            total_items += len(item_tracks)

            def assoc_person_name_by_iou(person_tracks: Dict[int, Track], face_dets: List[Dict], iou_thr=0.05):
                mapping = {}
                for tid, t in person_tracks.items():
                    best = ("Unknown", -1.0, 0.0)
                    for fd in face_dets:
                        fb = fd.get("bbox_xyxy")
                        if not fb: continue
                        ii = iou(t.bbox, tuple(map(int, fb)))
                        if ii > best[2] and ii >= iou_thr:
                            best = (fd.get("person_id","Unknown"), float(fd.get("similarity", -1.0)), ii)
                    mapping[tid] = best
                return mapping
            person_name_by_track = assoc_person_name_by_iou(person_tracks, face_dets, iou_thr=0.05)

            item_class_by_track: Dict[int, str] = {}
            for tid, t in item_tracks.items():
                best = None; best_iou = 0.0; best_conf = 0.0
                for it in item_dets:
                    ii = iou(t.bbox, tuple(map(int, it["bbox_xyxy"])))
                    if ii > best_iou:
                        best_iou = ii; best = it; best_conf = float(it.get("score", 0.0))
                if best is not None and best_iou >= 0.05:
                    cname = best.get("class_name", "unknown")
                    item_label_hist[tid][cname] += 1
                    last_conf_by_track[tid] = best_conf
                    gate.update_buffers(tid, t_s, cname, best_conf)
                    item_class_by_track[tid] = max(item_label_hist[tid].items(), key=lambda x: x[1])[0]
                else:
                    item_class_by_track[tid] = item_class_by_track.get(tid, "unknown")

            # 상체 기준선/포인트 표시
            for t in person_tracks.values():
                x1,y1,x2,y2 = t.bbox
                upper_y = int(y1 + (y2-y1)*UPPER_RATIO)
                cv2.line(frame, (x1, upper_y), (x2, upper_y), (0,255,255), 1)
            for t in item_tracks.values():
                cx = (t.bbox[0]+t.bbox[2])//2; cy = (t.bbox[1]+t.bbox[3])//2
                cv2.circle(frame, (cx,cy), 4, (255,255,0), -1)

            # holding 페어 계산
            holding_pairs = []
            for pid, pt in person_tracks.items():
                for iid, it in item_tracks.items():
                    cond_upper = is_item_in_upper_body(pt.bbox, it.bbox, upper_ratio=UPPER_RATIO)
                    cond_iou   = (iou(pt.bbox, it.bbox) >= IOU_MIN)
                    if cond_upper and cond_iou:
                        holding_pairs.append((pid, iid, pt.bbox, it.bbox, cond_upper, cond_iou))

            # 에러 평가 (2.0s 지속)
            has_face = len(face_dets) > 0
            has_item = len(item_tracks) > 0
            if no_face_gate.evaluate(condition=(not has_face and has_item)):
                cue = {
                    "ts_ms": int(t_s*1000),
                    "type": "ERROR", "reason": "NO_FACE",
                    "detail": f"faces={len(face_dets)}, items={len(item_tracks)}, holding_pairs={len(holding_pairs)}"
                }
                ui_f.write(json.dumps(cue, ensure_ascii=False) + "\n"); ui_f.flush()
            if no_item_gate.evaluate(condition=(has_face and not has_item)):
                cue = {
                    "ts_ms": int(t_s*1000),
                    "type": "ERROR", "reason": "NO_ITEM",
                    "detail": f"faces={len(face_dets)}, items={len(item_tracks)}, holding_pairs={len(holding_pairs)}"
                }
                ui_f.write(json.dumps(cue, ensure_ascii=False) + "\n"); ui_f.flush()

            # === 2초 누적 시 이벤트 발화 ===
            current_pairs = set((pid, iid) for pid, iid, *_ in holding_pairs)
            for pid, iid, pbox, ibox, cu, ci in holding_pairs:
                key = (pid, iid)
                # 시작 시각/지속 관리
                if key not in hold_started_at:
                    hold_started_at[key] = t_s
                dur = t_s - hold_started_at[key]
                if dur >= 2.0 and not fired[key]:
                    on_hold_event(pid=pid, iid=iid, t_s=t_s, duration_s=dur, frame=frame,
                                  person_box=pbox, item_box=ibox,
                                  person_name_by_track=person_name_by_track, item_class_by_track=item_class_by_track,
                                  cond_upper=cu, cond_iou=ci)
                    fired[key] = True
            # 끊김 처리(0.5s)
            for key in list(hold_started_at.keys()):
                if key not in current_pairs:
                    if key not in last_seen_at:
                        last_seen_at[key] = t_s
                    if (t_s - last_seen_at.get(key, t_s)) >= 0.5:
                        hold_started_at.pop(key, None); last_seen_at.pop(key, None); fired[key] = False
                else:
                    last_seen_at[key] = t_s

            # 클래스 잠금 해제
            person_has_item = defaultdict(lambda: False)
            for pid, iid, *_ in holding_pairs: person_has_item[pid] = True
            for pid in person_tracks.keys(): gate.release_if_needed(pid, t_s, has_any_item=person_has_item[pid])

            # 프레임 로그
            rec = {
                "timestamp_ms": int(t_s*1000), "frame_idx": int(frame_idx),
                "faces": face_dets, "items": item_dets,
                "tracks": {
                    "persons": [{"track_id": tid, "bbox": [int(v) for v in t.bbox],
                                 "name": (lambda nm: nm[0])(person_name_by_track.get(tid, ("Unknown",-1.0,0.0)))} for tid, t in person_tracks.items()],
                    "items":   [{"track_id": tid, "bbox": [int(v) for v in t.bbox],
                                 "class_name": (lambda ic: ic)(item_class_by_track.get(tid, "unknown"))} for tid, t in item_tracks.items()]
                },
                "meta": {"yolo_conf": yolo_conf, "yolo_iou": yolo_iou, "face_similarity_thr": face_thr}
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); writer.write(frame)

    finally:
        cap.release(); writer.release()
        for f in [jf, event_f, diag_f, ui_f, gate_f]:
            try: f.close()
            except Exception: pass

    summary = {
        "video": str(video_path), "fps_assumed": fps, "frames_processed": frame_idx,
        "faces_counted": total_faces, "avg_items_tracked_per_frame": round(total_items / max(1, frame_idx), 3),
        "outputs": {
            "overlay_video": str(out_video),
            "frame_log_jsonl": str(out_jsonl),
            "event_log_jsonl": str(out_events),
            "holding_diag_jsonl": str(out_holding),
            "snapshots_dir": str(out_snaps),
            "ui_cues_jsonl": str(out_ui_cues),
            "gate_log_jsonl": str(out_gate),
            "corrections_jsonl": str(OUTPUT_ROOT / f"{base}_corrections.jsonl")
        },
        "params": {
            "hold_s": 2.0, "release_s": 0.5, "upper_ratio": UPPER_RATIO, "iou_hold_min": IOU_MIN,
            "class_stable_window": CLASS_STABLE_WINDOW_S, "agreement_ratio_min": AGREEMENT_RATIO_MIN,
            "cooldown_s": COOLDOWN_S, "no_item_window_s": NO_ITEM_WINDOW_S,
            "yolo_conf": yolo_conf, "tracker_iou_thresh": 0.25, "time_source": "frame_idx/fps"
        }
    }
    with open(out_summary, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print("\n✅ 완료")
    print("   - 오버레이:", out_video)
    print("   - 프레임로그:", out_jsonl)
    print("   - 이벤트   :", out_events)
    print("   - UI Cues  :", out_ui_cues)
    print("   - 요약     :", out_summary)

if __name__ == "__main__":
    main()
