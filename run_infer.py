#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict, Counter
from datetime import datetime

import cv2
import numpy as np

# --- YOLO (Ultralytics) ---
from ultralytics import YOLO as UL_YOLO
# --- InsightFace ---
from insightface.app import FaceAnalysis

# ======================== 고정 경로 =========================
YOLO_WEIGHT = Path(r"C:\Users\User\Desktop\final\models\yolo\detector.pt")
CLASSES_TXT = Path(r"C:\Users\User\Desktop\final\config\classes.txt")
FACE_BANK_DIR = Path(r"C:\Users\User\Desktop\final\face_bank")

# ✅ 상위 출력 루트는 고정, 하위 폴더명은 런타임에 입력받아 생성
BASE_OUTPUT_ROOT = Path(r"C:\Users\User\Desktop\csv\output")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

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

def load_face_bank(app: FaceAnalysis, bank_root: Optional[Path]):
    vecs = []
    if not bank_root or not bank_root.exists():
        return vecs
    for pdir in sorted([d for d in bank_root.iterdir() if d.is_dir()]):
        pid = pdir.name
        mean_p = pdir / "mean_embedding.npy"
        if mean_p.exists():
            emb = np.load(str(mean_p))
            emb = emb.mean(axis=0) if emb.ndim > 1 else emb
            vecs.append((pid, emb.astype(np.float32)))
            continue
        imgs = list(pdir.glob("*.jpg")) + list(pdir.glob("*.jpeg")) + list(pdir.glob("*.png"))
        imgs = imgs[:20]
        tmp = []
        for ip in imgs:
            img = cv2.imread(str(ip))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            f0 = faces[0]
            emb = getattr(f0, "normed_embedding", None)
            if emb is None:
                emb = getattr(f0, "embedding", None)
            if emb is None:
                continue
            tmp.append(np.asarray(emb, dtype=np.float32))
        if tmp:
            vecs.append((pid, np.mean(np.stack(tmp, axis=0), axis=0)))
    return vecs

# ---------- 출력 하위 폴더명 보정 ----------
def sanitize_folder_name(name: str) -> str:
    """Windows에서 안전한 폴더명으로 정리(한글/영문/숫자/공백/.-_ 만 허용). 연속 공백은 하나로."""
    name = name.strip()
    if not name:
        return ""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', name)  # 금지 문자 제거
    name = re.sub(r'\s+', ' ', name)  # 연속 공백 정리
    return name[:100]  # 너무 길면 자름

def next_nonconflicting_dir(base_dir: Path) -> Path:
    """이미 존재하면 _1, _2 ... 붙여서 충돌 없는 폴더 경로 리턴"""
    if not base_dir.exists():
        return base_dir
    i = 1
    while True:
        cand = base_dir.parent / f"{base_dir.name}_{i}"
        if not cand.exists():
            return cand
        i += 1

# ===================== 트래커 & 홀드 엔진 =======================
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
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    return float(inter / max(1e-6, (area_a + area_b - inter)))

class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.25, max_miss: int = 30):
        self.iou_thresh = iou_thresh
        self.max_miss = max_miss
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[Tuple[int,int,int,int, float]], frame_idx: int) -> Dict[int, Track]:
        det_bboxes = [d[:4] for d in dets]
        det_scores = [d[4] for d in dets]
        tids = list(self.tracks.keys())
        tbxs = [self.tracks[t].bbox for t in tids]

        iou_mat = np.zeros((len(tbxs), len(det_bboxes)), dtype=np.float32)
        for i_, tb in enumerate(tbxs):
            for j_, db in enumerate(det_bboxes):
                iou_mat[i_, j_] = iou(tb, db)

        matched_t, matched_d = set(), set()
        pairs = []
        for _ in range(min(len(tbxs), len(det_bboxes))):
            i_, j_ = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[i_, j_] < self.iou_thresh:
                break
            if i_ in matched_t or j_ in matched_d:
                iou_mat[i_, j_] = -1
                continue
            matched_t.add(i_); matched_d.add(j_)
            pairs.append((tids[i_], j_))
            iou_mat[i_, :] = -1; iou_mat[:, j_] = -1

        for tid, j_ in pairs:
            self.tracks[tid].bbox = tuple(map(int, det_bboxes[j_]))
            self.tracks[tid].last_seen = frame_idx
            self.tracks[tid].hits += 1
            self.tracks[tid].miss = 0
            self.tracks[tid].score = float(det_scores[j_])

        for j_, db in enumerate(det_bboxes):
            if j_ not in matched_d:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(tid, tuple(map(int, db)), frame_idx, 1, 0, float(det_scores[j_]))

        to_del = []
        for tid in list(self.tracks.keys()):
            if tid not in [t for t, _ in pairs]:
                self.tracks[tid].miss += 1
            if self.tracks[tid].miss > self.max_miss:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return self.tracks

def box_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def is_item_in_upper_body(person_box, item_box, upper_ratio: float = 0.85) -> bool:
    px1,py1,px2,py2 = person_box
    cx, cy = box_center(item_box)
    upper_y = int(py1 + (py2 - py1) * upper_ratio)
    return (px1 <= cx <= px2) and (py1 <= cy <= upper_y)

def is_holding(person_box, item_box) -> bool:
    return is_item_in_upper_body(person_box, item_box, upper_ratio=0.85) or (iou(person_box, item_box) >= 0.05)

class HoldEventEngine:
    def __init__(self, hold_s=2.0, release_s=0.5, on_event: Optional[Callable]=None, diag_writer=None):
        self.hold_s = hold_s
        self.release_s = release_s
        self.on_event = on_event or (lambda *args, **kwargs: None)
        self.hold_started_at: Dict[Tuple[int,int], float] = {}
        self.last_hold_seen_at: Dict[Tuple[int,int], float] = {}
        self.fired: Dict[Tuple[int,int], bool] = defaultdict(lambda: False)
        self.diag_writer = diag_writer

    def update(self, person_tracks: Dict[int, Track], item_tracks: Dict[int, Track],
               t_s: float, frame, cb_extra: Dict):
        holding = []
        for pid, pt in person_tracks.items():
            for iid, it in item_tracks.items():
                cond_upper = is_item_in_upper_body(pt.bbox, it.bbox, upper_ratio=0.85)
                cond_iou   = (iou(pt.bbox, it.bbox) >= 0.05)
                if cond_upper or cond_iou:
                    holding.append((pid, iid, pt.bbox, it.bbox, cond_upper, cond_iou))

        current = {(pid, iid) for pid, iid, *_ in holding}

        for pid, iid, pbox, ibox, cond_upper, cond_iou in holding:
            key = (pid, iid)
            self.last_hold_seen_at[key] = t_s
            if key not in self.hold_started_at:
                self.hold_started_at[key] = t_s
            dur = t_s - self.hold_started_at[key]

            if self.diag_writer is not None:
                self.diag_writer.write(json.dumps({
                    "timestamp_s": t_s,
                    "pair": {"person_id": pid, "item_id": iid},
                    "hold_duration_s": round(dur, 3),
                    "cond_upper": bool(cond_upper),
                    "cond_iou_ge_0p05": bool(cond_iou),
                    "fired": bool(self.fired[key])
                }, ensure_ascii=False) + "\n")

            if not self.fired[key] and dur >= self.hold_s:
                self.fired[key] = True
                self.on_event(pid=pid, iid=iid, t_s=t_s, duration_s=dur,
                              frame=frame, person_box=pbox, item_box=ibox, **cb_extra)

        for key in list(self.hold_started_at.keys()):
            if key not in current:
                last = self.last_hold_seen_at.get(key, t_s)
                if t_s - last >= self.release_s:
                    self.hold_started_at.pop(key, None)
                    self.last_hold_seen_at.pop(key, None)
                    self.fired[key] = False

# ========== 보조: 트랙별 이름/클래스 안정화 ==========
def assoc_person_name_by_iou(person_tracks: Dict[int, Track], face_dets: List[Dict], iou_thr=0.05):
    mapping = {}  # track_id -> (name, similarity, iou)
    for tid, t in person_tracks.items():
        best = ("Unknown", -1.0, 0.0)
        for fd in face_dets:
            fb = fd.get("bbox_xyxy")
            if not fb:
                continue
            i = iou(t.bbox, tuple(map(int, fb)))
            if i > best[2] and i >= iou_thr:
                best = (fd.get("person_id","Unknown"), float(fd.get("similarity", -1.0)), i)
        mapping[tid] = best
    return mapping

# =========================== 메인 ===============================
def main():
    print("=== Unified Inference (YOLO items + InsightFace faces) — v2.3.1 ===")

    # ✅ 출력 하위 폴더명 입력
    BASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sub = input(f"결과물 하위 폴더 이름을 입력하세요 (상위: {BASE_OUTPUT_ROOT}): ").strip()
    sub = sanitize_folder_name(sub)
    if not sub:
        # 비었으면 타임스탬프 기본명
        sub = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    OUTPUT_ROOT = BASE_OUTPUT_ROOT / sub
    OUTPUT_ROOT = next_nonconflicting_dir(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"▶ 결과물은 여기 저장됩니다: {OUTPUT_ROOT}")

    # 1) 테스트 영상 경로
    video_path_in = input("테스트할 영상의 전체 경로를 입력하세요: ").strip().strip('"')
    if not video_path_in:
        print("❌ 영상 경로가 비었습니다."); sys.exit(2)
    video_path = Path(video_path_in).resolve()
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}"); sys.exit(2)
    if video_path.suffix.lower() not in VIDEO_EXTS:
        print(f"⚠️ 비권장 확장자({video_path.suffix}). 계속 진행합니다.")

    # 2) 출력 경로 구성 (선택한 OUTPUT_ROOT 하위에 저장)
    base = video_path.stem
    out_video    = OUTPUT_ROOT / f"{base}_result.mp4"
    out_jsonl    = OUTPUT_ROOT / f"{base}_result.jsonl"
    out_events   = OUTPUT_ROOT / f"{base}_events.jsonl"
    out_snaps    = OUTPUT_ROOT / f"{base}_snaps"
    out_holding  = OUTPUT_ROOT / f"{base}_holding_log.jsonl"
    out_summary  = OUTPUT_ROOT / f"{base}_summary.json"
    out_ui_cues  = OUTPUT_ROOT / f"{base}_ui_cues.jsonl"
    out_snaps.mkdir(parents=True, exist_ok=True)

    # 3) 임계치
    yolo_conf = 0.30
    yolo_iou  = 0.50
    face_thr  = 0.40

    # 4) 리소스 체크/로드
    if not YOLO_WEIGHT.exists():
        print(f"❌ YOLO 가중치를 찾을 수 없습니다: {YOLO_WEIGHT}"); sys.exit(1)

    class_names = load_classes(CLASSES_TXT if CLASSES_TXT.exists() else None)
    if class_names:
        print(f"ℹ️ classes.txt 로드: {len(class_names)} classes")
    else:
        print("ℹ️ classes.txt 없음/비어있음 → 클래스명 없이 진행")

    yolo = UL_YOLO(str(YOLO_WEIGHT))
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640,640))  # 0=GPU, -1=CPU

    try:
        import torch, onnxruntime as ort
        print("[DIAG] torch.cuda.is_available:", torch.cuda.is_available())
        print("[DIAG] ORT providers:", ort.get_available_providers())
        try:
            dev = next(yolo.model.parameters()).device
        except Exception:
            dev = "unknown"
        print("[DIAG] YOLO model device:", dev)
    except Exception as e:
        print("[DIAG] 진단 로그 실패:", e)

    gallery = load_face_bank(app, FACE_BANK_DIR) if FACE_BANK_DIR and FACE_BANK_DIR.exists() else []
    if gallery:
        print(f"ℹ️ face_bank 로드: {len(gallery)}명")
    else:
        print("ℹ️ face_bank 없음/비어있음 → 얼굴은 ID=Unknown으로 표시")

    # 5) 비디오 입출력
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다."); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = ensure_writer(out_video, w, h, fps)
    jf = out_jsonl.open("w", encoding="utf-8")
    event_f = out_events.open("w", encoding="utf-8")
    diag_f = out_holding.open("w", encoding="utf-8")
    ui_f = out_ui_cues.open("w", encoding="utf-8")

    # 6) 트래커/엔진
    person_tracker = SimpleTracker(iou_thresh=0.25, max_miss=int(2*fps))
    item_tracker   = SimpleTracker(iou_thresh=0.25, max_miss=int(2*fps))
    item_label_hist: Dict[int, Counter] = defaultdict(Counter)

    def on_hold_event(pid, iid, t_s, duration_s, frame, person_box, item_box,
                      event_writer, snap_dir, ui_writer,
                      person_name_by_track, item_class_by_track):
        x1,y1,x2,y2 = map(int, item_box)
        H, W = frame.shape[:2]
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(x1+1, min(W, x2)); y2 = max(y1+1, min(H, y2))
        crop = frame[y1:y2, x1:x2].copy()
        snap_name = f"event_p{pid}_i{iid}_{int(t_s*1000)}.jpg"
        cv2.imwrite(str(snap_dir / snap_name), crop)

        pname, psim, _ = person_name_by_track.get(pid, ("Unknown", -1.0, 0.0))
        iname = item_class_by_track.get(iid, "Unknown")

        ui_hint = None
        if str(pname).lower() == "unknown":
            ui_hint = "카메라를 정면으로 바라보고 다시 서주세요"
            ui_writer.write(json.dumps({
                "ts_s": float(t_s),
                "type": "prompt",
                "level": "info",
                "target": {"person_track_id": int(pid)},
                "message": ui_hint,
                "reason": "unknown_face_on_event"
            }, ensure_ascii=False) + "\n")
            ui_writer.flush()

        evt = {
            "timestamp_s": float(t_s),
            "duration_s": float(duration_s),
            "person_track_id": int(pid),
            "item_track_id": int(iid),
            "person_box": [int(v) for v in person_box],
            "item_box": [int(v) for v in item_box],
            "person_name": pname,
            "person_similarity": float(psim),
            "item_class": iname,
            "status": "needs_face_front" if str(pname).lower() == "unknown" else "pending",
            "ui_hint": ui_hint,
            "snapshot": snap_name
        }
        event_writer.write(json.dumps(evt, ensure_ascii=False) + "\n")
        event_writer.flush()

        print(f"[EVENT] HOLD≥{2.0:.1f}s @ {t_s:.2f}s  person:{pid}({pname})  item:{iid}({iname})  dur:{duration_s:.2f}s")

    hold_engine = HoldEventEngine(
        hold_s=2.0, release_s=0.5,
        on_event=lambda **kw: on_hold_event(
            **kw,
            event_writer=event_f,
            snap_dir=out_snaps,
            ui_writer=ui_f,
            person_name_by_track=person_name_by_track,   # 람다에서만 전달
            item_class_by_track=item_class_by_track
        ),
        diag_writer=diag_f
    )

    print(f"\n▶ 시작: {video_path.name}  {w}x{h} @ {fps:.1f}fps")
    print(f"▶ 출력 폴더: {OUTPUT_ROOT}\n")

    frame_idx = 0
    total_faces = 0
    total_items = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            t_s = frame_idx / fps

            # --- YOLO ---
            try:
                yres = yolo.predict(
                    source=frame, conf=yolo_conf, iou=yolo_iou,
                    device=0, half=True, verbose=False
                )[0]
            except Exception:
                yres = yolo.predict(
                    source=frame, conf=yolo_conf, iou=yolo_iou,
                    device=0, half=False, verbose=False
                )[0]

            item_dets = []
            yolo_person_dets = []
            if yres and yres.boxes is not None:
                boxes = yres.boxes.xyxy.cpu().numpy()
                scores = yres.boxes.conf.cpu().numpy()
                clss   = yres.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(boxes, scores, clss):
                    cname = str(c)
                    if class_names is not None and 0 <= c < len(class_names):
                        cname = class_names[c]
                    if cname.lower() == "person":
                        yolo_person_dets.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s)))
                        draw_box(frame, b, (0, 200, 200), f"person {s:.2f}")
                    else:
                        item_dets.append({
                            "bbox_xyxy": [float(v) for v in b],
                            "score": float(s),
                            "class_id": int(c),
                            "class_name": cname
                        })
                        draw_box(frame, b, (60,180,255), f"{cname} {s:.2f}")

            # --- InsightFace ---
            face_dets = []
            faces = app.get(frame)
            for f in faces:
                x1,y1,x2,y2 = map(float, f.bbox)
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)
                pid, sim = "Unknown", -1.0
                if emb is not None and gallery:
                    best_id, best_sim = "Unknown", -1.0
                    for gid, gvec in gallery:
                        s = cosine_sim(emb, gvec)
                        if s > best_sim:
                            best_sim, best_id = s, gid
                    if best_sim >= face_thr:
                        pid, sim = best_id, best_sim

                face_dets.append({
                    "bbox_xyxy": [x1,y1,x2,y2],
                    "person_id": pid,
                    "similarity": float(sim)
                })
                label = f"{pid} {sim:.2f}" if pid!="Unknown" else "Unknown"
                draw_box(frame, (x1,y1,x2,y2), (0,200,0) if pid!="Unknown" else (0,100,255), label)
            total_faces += len(face_dets)

            # --- 사람 입력 박스 ---
            person_inputs: List[Tuple[int,int,int,int, float]] = []
            if yolo_person_dets:
                person_inputs = yolo_person_dets
            else:
                H, W = frame.shape[:2]
                for fd in face_dets:
                    fx1, fy1, fx2, fy2 = map(int, fd["bbox_xyxy"])
                    fw, fh = fx2 - fx1, fy2 - fy1
                    if fw <= 0 or fh <= 0:
                        continue
                    scale = 3.0
                    px1 = int(fx1 - 0.5*fw)
                    px2 = int(fx2 + 0.5*fw)
                    py1 = int(fy1 - 0.5*fh)
                    py2 = int(fy2 + scale*fh)
                    px1 = max(0, px1); py1 = max(0, py1)
                    px2 = min(W, px2);  py2 = min(H, py2)
                    if px2 - px1 < 2 or py2 - py1 < 2:
                        continue
                    person_inputs.append((px1, py1, px2, py2, 0.80))
                    cv2.rectangle(frame, (px1,py1), (px2,py2), (0,255,255), 1)
                    cv2.putText(frame, "upper-body est.", (px1, max(0, py1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

            # --- 트래킹 ---
            item_inputs = []
            for it in item_dets:
                x1,y1,x2,y2 = map(int, it["bbox_xyxy"])
                sc = float(it["score"])
                item_inputs.append((x1,y1,x2,y2,sc))

            person_tracks = person_tracker.update(person_inputs, frame_idx)
            item_tracks   = item_tracker.update(item_inputs, frame_idx)
            total_items += len(item_tracks)

            # --- 트랙 라벨 안정화 ---
            person_name_by_track = assoc_person_name_by_iou(person_tracks, face_dets, iou_thr=0.05)

            for tid, t in item_tracks.items():
                best = None; best_iou = 0.0
                for it in item_dets:
                    i = iou(t.bbox, tuple(map(int, it["bbox_xyxy"])))
                    if i > best_iou:
                        best_iou = i; best = it
                if best is not None and best_iou >= 0.05:
                    cname = best.get("class_name", "Unknown")
                    item_label_hist[tid][cname] += 1

            item_class_by_track: Dict[int, str] = {}
            for tid, hist in item_label_hist.items():
                item_class_by_track[tid] = max(hist.items(), key=lambda x: x[1])[0] if hist else "Unknown"

            # --- 디버그 오버레이 ---
            for t in person_tracks.values():
                x1,y1,x2,y2 = t.bbox
                upper_y = int(y1 + (y2-y1)*0.85)
                cv2.line(frame, (x1, upper_y), (x2, upper_y), (0,255,255), 1)
            for t in item_tracks.values():
                cx = (t.bbox[0]+t.bbox[2])//2
                cy = (t.bbox[1]+t.bbox[3])//2
                cv2.circle(frame, (cx,cy), 4, (255,255,0), -1)

            # --- 홀드 감지(2s) & 이벤트 발행 ---
            hold_engine.update(
                person_tracks=person_tracks,
                item_tracks=item_tracks,
                t_s=t_s,
                frame=frame,
                cb_extra={}  # 중복 전달 방지
            )

            # --- 프레임 로그(JSONL) ---
            rec = {
                "timestamp_ms": int(t_s*1000),
                "frame_idx": int(frame_idx),
                "faces": face_dets,
                "items": item_dets,
                "tracks": {
                    "persons": [
                        {
                            "track_id": tid,
                            "bbox": [int(v) for v in t.bbox],
                            "name": person_name_by_track.get(tid, ("Unknown",-1.0,0.0))[0]
                        } for tid, t in person_tracks.items()
                    ],
                    "items": [
                        {
                            "track_id": tid,
                            "bbox": [int(v) for v in t.bbox],
                            "class_name": item_class_by_track.get(tid, "Unknown")
                        } for tid, t in item_tracks.items()
                    ]
                },
                "meta": {
                    "yolo_conf": yolo_conf,
                    "yolo_iou": yolo_iou,
                    "face_similarity_thr": face_thr
                }
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        jf.close()
        event_f.close()
        diag_f.close()
        ui_f.close()

    summary = {
        "video": str(video_path),
        "fps_assumed": fps,
        "frames_processed": frame_idx,
        "faces_counted": total_faces,
        "avg_items_tracked_per_frame": round(total_items / max(1, frame_idx), 3),
        "outputs": {
            "overlay_video": str(out_video),
            "frame_log_jsonl": str(out_jsonl),
            "event_log_jsonl": str(out_events),
            "holding_diag_jsonl": str(out_holding),
            "snapshots_dir": str(out_snaps),
            "ui_cues_jsonl": str(out_ui_cues)
        },
        "params": {
            "hold_s": 2.0,
            "release_s": 0.5,
            "upper_ratio": 0.85,
            "iou_hold_min": 0.05,
            "yolo_conf": 0.30,
            "tracker_iou_thresh": 0.25,
            "tracker_max_miss_frames": int(2*fps),
            "time_source": "frame_idx/fps"
        }
    }
    with out_summary.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print("\n✅ 완료")
    print("   - 오버레이:", out_video)
    print("   - 프레임로그:", out_jsonl)
    print("   - 이벤트   :", out_events, "(이벤트 순간 라벨 포함)")
    print("   - 홀드진단 :", out_holding)
    print("   - 스냅샷   :", out_snaps)
    print("   - 요약     :", out_summary)
    print("   - UI큐     :", out_ui_cues, "(Unknown 안내)")

if __name__ == "__main__":
    main()

