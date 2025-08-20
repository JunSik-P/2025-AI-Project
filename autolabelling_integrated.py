# auto_labeling.py
# 단일 클래스 반자동 라벨링 파이프라인 (YOLO11/YOLOv8 호환)
# - 샘플 라벨이 있는 이미지들만으로 학습
# - 라벨 없는 이미지에 대해서만 자동 라벨링
# - 기존 수작업 라벨은 절대 덮어쓰지 않음
# - 예외(0개/2개 이상) 라벨 박스 개수 리포트
# - 마지막에 labels/<class> 폴더에 txt와 "짝" 이미지 복사 보장
# - Windows 안전: __main__ 가드 + workers=0

import os
import glob
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# ============ 사용자 입력 ============
PROJECT_DIR   = Path(r"C:\Users\user\Desktop\project")  # project 폴더 경로
TARGET_CLASS  = "bulletproof_plate"                     # 작업할 클래스
IMG_EXTS      = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# 학습/추론 하이퍼파라미터
EPOCHS  = 50
IMGSZ   = 640
BATCH   = 16
SEED    = 42
CONF    = 0.25
IOU     = 0.6
DEVICE  = 0         # GPU 없으면 'cpu'
# ====================================


def _list_images(dir_path: Path):
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])


def _copy_with_label(img_list, images_dst: Path, labels_dst: Path, labels_src: Path):
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    for img in img_list:
        stem = img.stem
        lbl = labels_src / f"{stem}.txt"}
        shutil.copy2(img, images_dst / img.name)
        if lbl.exists():  # 라벨 있으면 같이 복사
            shutil.copy2(lbl, labels_dst / lbl.name)


def _write_yaml(dataset_root: Path, cname: str):
    yml = {
        "train": str((dataset_root / "images/train").resolve()),
        "val":   str((dataset_root / "images/val").resolve()),
        "nc": 1,
        "names": [cname],
    }
    (dataset_root / "data.yaml").write_text(yaml.dump(yml, allow_unicode=True), encoding="utf-8")


def _find_image_for_stem(images_dir: Path, stem: str):
    # images/<class>/ 에서 같은 stem의 이미지를 확장자 우선순위대로 탐색
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # 혹시 대소문자/확장자 섞인 경우를 위해 광역 검색(최후 수단)
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem:
            return p
    return None


def main():
    # 경로 설정
    images_path       = PROJECT_DIR / "images" / TARGET_CLASS
    labels_path       = PROJECT_DIR / "labels" / TARGET_CLASS
    dataset_tmp_root  = PROJECT_DIR / "dataset_tmp"
    dataset_tmp_class = dataset_tmp_root / f"dataset_tmp_{TARGET_CLASS}"

    # 임시 학습 폴더 초기화
    if dataset_tmp_class.exists():
        shutil.rmtree(dataset_tmp_class)
    (dataset_tmp_class / "images/train").mkdir(parents=True, exist_ok=True)
    (dataset_tmp_class / "images/val").mkdir(parents=True, exist_ok=True)
    (dataset_tmp_class / "labels/train").mkdir(parents=True, exist_ok=True)
    (dataset_tmp_class / "labels/val").mkdir(parents=True, exist_ok=True)

    # ---------- 1) 학습용 샘플 수집: "라벨이 있는 이미지"만 ----------
    all_imgs = _list_images(images_path)
    labeled_imgs = []
    for img in all_imgs:
        if (labels_path / f"{img.stem}.txt").exists():
            labeled_imgs.append(img)

    if len(labeled_imgs) == 0:
        print(f"[!] 샘플 라벨이 하나도 없습니다: {labels_path}")
        return

    # train/val 분할 (라벨이 있는 이미지들만)
    random.Random(SEED).shuffle(labeled_imgs)
    split_idx = max(1, int(0.8 * len(labeled_imgs)))
    train_images = labeled_imgs[:split_idx]
    val_images   = labeled_imgs[split_idx:]

    _copy_with_label(train_images, dataset_tmp_class / "images/train", dataset_tmp_class / "labels/train", labels_path)
    _copy_with_label(val_images,   dataset_tmp_class / "images/val",   dataset_tmp_class / "labels/val",   labels_path)

    # ---------- 2) data.yaml 생성 ----------
    _write_yaml(dataset_tmp_class, TARGET_CLASS)
    print(f"[INFO] YOLO dataset created at {dataset_tmp_class}")

    # ---------- 3) YOLO 학습 ----------
    model = YOLO("yolo11n.pt")  # (또는 "yolov8n.pt")
    model.train(
        data=str(dataset_tmp_class / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,          # ★ Windows 안전모드
        device=DEVICE,
        seed=SEED,
        verbose=True,
    )
    best = Path(model.trainer.best)

    # ---------- 4) 라벨 없는 이미지 목록 구하기 ----------
    labeled_stems = {p.stem for p in labeled_imgs}
    unlabeled_imgs = [img for img in all_imgs if img.stem not in labeled_stems]

    # ---------- 5) 자동 라벨링 (라벨 없는 것만) ----------
    if len(unlabeled_imgs) > 0:
        model = YOLO(best)  # 학습된 가중치 로드
        results = model.predict(
            source=[str(p) for p in unlabeled_imgs],
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            save=False,
            verbose=False,
            device=DEVICE,
            stream=False,
        )

        # ---------- 6) 라벨 저장 + 박스 개수 리포트 ----------
        problems = []  # (filename, count)
        for r in results:
            img_path = Path(r.path)
            stem = img_path.stem
            out_txt = labels_path / f"{stem}.txt"

            # 안전: 기존 수작업 라벨이 있으면 건너뜀
            if out_txt.exists():
                continue

            # 박스가 없을 수도 있음 (그 경우에도 빈 파일을 만들어 1:1 대응 유지)
            lines = []
            if r.boxes is not None and len(r.boxes) > 0:
                # xywhn: normalized [0,1]
                for xywh in r.boxes.xywhn.cpu().tolist():
                    x, y, w, h = xywh
                    lines.append(f"0 {x} {y} {w} {h}")
            # else: 검출 0개 → 빈 txt 생성

            out_txt.write_text("\n".join(lines), encoding="utf-8")

            # 리포트: 박스 개수가 1개가 아니면 기록
            if len(lines) != 1:
                problems.append((out_txt.name, len(lines)))

        print(f"[✓] 자동 라벨링 완료 → 저장 위치: {labels_path}")

        if problems:
            print("\n[!] 라벨 박스 개수가 1개가 아닌 파일 목록:")
            for fname, cnt in problems[:100]:  # 너무 많으면 100개까지만 표시
                print(f"  - {fname}: {cnt} boxes")
            if len(problems) > 100:
                print(f"  ... 외 {len(problems)-100}개 더 있음")
        else:
            print("\n[INFO] 모든 자동 라벨링 결과가 정확히 1개 박스입니다. ✅")
    else:
        print("[INFO] 라벨 없는 이미지가 없습니다. 자동 라벨링 생략.")

    # ---------- 7) labels/<class> 폴더에 "짝" 이미지 채워 넣기 ----------
    # 목적: labels/<class>/ 안에 존재하는 모든 .txt 와 동일한 stem 의 이미지를
    #       images/<class>/ 에서 찾아 labels/<class>/ 로 '복사' (이미 존재하면 스킵)
    txt_files = sorted([p for p in labels_path.glob("*.txt")])
    copied, missing = 0, []
    for txt in txt_files:
        stem = txt.stem
        # labels/<class>/ 에 이미지가 이미 있는지 확인
        already = None
        for ext in IMG_EXTS:
            cand = labels_path / f"{stem}{ext}"
            if cand.exists():
                already = cand
                break
        if already:
            continue  # 이미 있음

        img_src = _find_image_for_stem(images_path, stem)
        if img_src is None:
            missing.append(stem)
            continue

        shutil.copy2(img_src, labels_path / img_src.name)
        copied += 1

    print(f"[✓] 라벨 폴더 이미지 정리: 새로 복사 {copied}개")
    if missing:
        print(f"[!] 짝 이미지 못 찾은 라벨 {len(missing)}개 (images/{TARGET_CLASS}에서 미발견)")
        preview = ", ".join(missing[:20])
        print(f"    예시: {preview}{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    # Windows 멀티프로세싱 안전 가드
    import multiprocessing as mp
    mp.freeze_support()
    main()
