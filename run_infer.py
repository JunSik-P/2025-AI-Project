#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# --- YOLO (Ultralytics) ---
from ultralytics import YOLO as UL_YOLO

# --- InsightFace ---
from insightface.app import FaceAnalysis

# ========== 유틸 ==========
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

def cosine_sim(a, b, eps=1e-9):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps)))

def draw_box(img, box, color, text=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if text:
        tw = max(80, 10 + 9*len(text))
        cv2.rectangle(img, (x1, y1-22), (x1+tw, y1), color, -1)
        cv2.putText(img, text, (x1+5, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def ensure_writer(out_path, w, h, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), (w, h))

def load_classes(classes_txt: Path):
    if not classes_txt.exists():
        return None
    names = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names if names else None

def load_face_bank(app: FaceAnalysis, bank_root: Path):
    """
    face_bank 구조:
      face_bank/
        kim/
          mean_embedding.npy  (있으면 우선)
          *.jpg, *.png ...    (없으면 이미지로 임베딩 평균 산출)
        park/
          ...
    반환: [(person_id, embedding_vector), ...]
    """
    vecs = []
    if not bank_root.exists():
        return vecs

    for pdir in sorted([d for d in bank_root.iterdir() if d.is_dir()]):
        pid = pdir.name
        mean_p = pdir / "mean_embedding.npy"
        if mean_p.exists():
            emb = np.load(str(mean_p))
            emb = emb.mean(axis=0) if emb.ndim > 1 else emb
            vecs.append((pid, emb.astype(np.float32)))
            continue

        # 이미지로부터 평균 임베딩 생성(최대 20장)
        imgs = list(pdir.glob("*.jpg")) + list(pdir.glob("*.jpeg")) + list(pdir.glob("*.png"))
        imgs = imgs[:20]
        tmp = []
        for ip in imgs:
            img = cv2.imread(str(ip))
            if img is None: continue
            faces = app.get(img)
            if not faces: continue
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

# ========== 메인 ==========
def main():
    print("=== Unified Inference (YOLO items + InsightFace faces) ===")

    # 1) 최종 폴더(final) 경로 입력
    final_root = input("최종 폴더(= final_pack 루트) 경로를 입력하세요: ").strip().strip('"')
    if not final_root:
        print("❌ 경로가 비었습니다."); sys.exit(2)
    final_root = Path(final_root).resolve()
    if not final_root.exists() or not final_root.is_dir():
        print(f"❌ 폴더를 찾을 수 없습니다: {final_root}"); sys.exit(2)

    # 2) 테스트 영상 경로 입력
    video_path = input("테스트할 영상의 전체 경로를 입력하세요: ").strip().strip('"')
    if not video_path:
        print("❌ 영상 경로가 비었습니다."); sys.exit(2)
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}"); sys.exit(2)
    if video_path.suffix.lower() not in VIDEO_EXTS:
        print(f"⚠️ 비권장 확장자({video_path.suffix}). 계속 진행합니다.")

    # 3) 출력 경로 구성: final/output/<파일명>_result.*
    output_dir = final_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = video_path.stem
    out_video = output_dir / f"{base}_result.mp4"
    out_jsonl = output_dir / f"{base}_result.jsonl"

    # 4) 필수 파일 경로 확인
    yolo_weight = final_root / "models" / "yolo" / "detector.pt"
    face_bank   = final_root / "face_bank"
    classes_txt = final_root / "config" / "classes.txt"

    missing = []
    if not yolo_weight.exists(): missing.append(str(yolo_weight))
    if not classes_txt.exists(): missing.append(str(classes_txt))
    if missing:
        print("❌ 필수 파일이 없습니다:")
        for m in missing: print("  -", m)
        sys.exit(1)

    # 5) 임계치 기본값
    yolo_conf = 0.45
    yolo_iou  = 0.50
    face_thr  = 0.40

    # 6) 클래스 로드
    class_names = load_classes(classes_txt)
    if not class_names:
        print("⚠️ classes.txt가 비어있거나 읽기 실패 → 클래스 이름 없이 진행합니다.")

    # 7) 모델 로드 (GPU 강제)
    # YOLO
    yolo = UL_YOLO(str(yolo_weight))
    try:
        yolo.to('cuda')  # ★ GPU 로드 강제
    except Exception as e:
        print("[YOLO] .to('cuda') 실패 → CPU로 계속:", e)

    # InsightFace (buffalo_l, ONNX Runtime CUDA 사용)
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider","CPUExecutionProvider"]  # ★ CUDA 우선
    )
    app.prepare(ctx_id=0, det_size=(640,640))  # ★ 0=GPU (강제), -1=CPU

    # 8) 진단 로그 (실제 백엔드 확인)
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

    # 9) 갤러리 임베딩 로드/생성
    gallery = load_face_bank(app, face_bank) if face_bank.exists() else []
    if gallery:
        print(f"ℹ️ face_bank 로드: {len(gallery)}명")
    else:
        print("ℹ️ face_bank 없음 또는 비어있음 → 얼굴은 ID=Unknown으로만 표시")

    # 10) 비디오 입출력
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다."); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = ensure_writer(out_video, w, h, fps)
    jf = out_jsonl.open("w", encoding="utf-8")

    print(f"\n▶ 시작: {video_path.name}  {w}x{h} @ {fps:.1f}fps")
    print(f"▶ 출력: {out_video.name}, {out_jsonl.name}\n")

    t0 = time.time(); frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ts_ms <= 0: ts_ms = int((time.time()-t0)*1000)

            # --- YOLO 추론 (GPU 강제) ---
            try:
                yres = yolo.predict(
                    source=frame,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    device=0,          # ★ GPU 0번 강제
                    half=True,         # 가능하면 FP16
                    verbose=False
                )[0]
            except Exception:
                # half 미지원/기타 이슈 시 FP32로 재시도
                yres = yolo.predict(
                    source=frame,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    device=0,
                    half=False,
                    verbose=False
                )[0]

            item_dets = []
            if yres and yres.boxes is not None:
                boxes = yres.boxes.xyxy.cpu().numpy()
                scores = yres.boxes.conf.cpu().numpy()
                clss   = yres.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(boxes, scores, clss):
                    cname = str(c)
                    if class_names is not None and 0 <= c < len(class_names):
                        cname = class_names[c]
                    item_dets.append({
                        "bbox_xyxy": [float(v) for v in b],
                        "score": float(s),
                        "class_id": int(c),
                        "class_name": cname
                    })
                    draw_box(frame, b, (60,180,255), f"{cname} {s:.2f}")

            # --- 얼굴 추론 + 갤러리 매칭 (GPU: ORT CUDA) ---
            face_dets = []
            faces = app.get(frame)
            for f in faces:
                x1,y1,x2,y2 = map(float, f.bbox)
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)

                pid, sim = "Unknown", -1.0
                if emb is not None and gallery:
                    # 최적 매칭
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

            # --- JSONL 기록 ---
            rec = {
                "timestamp_ms": ts_ms,
                "frame_idx": frame_idx,
                "faces": face_dets,
                "items": item_dets,
                "meta": {
                    "yolo_conf": yolo_conf,
                    "yolo_iou": yolo_iou,
                    "face_similarity_thr": face_thr
                }
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # --- 영상 기록 ---
            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        jf.close()

    dur = time.time() - t0
    fps_avg = frame_idx / max(dur, 1e-6)
    print(f"\n✅ 완료: {frame_idx} 프레임, {dur:.1f}s, 평균 {fps_avg:.2f} FPS")
    print(f"   - 오버레이: {out_video}")
    print(f"   - JSONL   : {out_jsonl}")

if __name__ == "__main__":
    main()
