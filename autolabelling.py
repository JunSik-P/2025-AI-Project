import os
from pathlib import Path
from ultralytics import YOLO
import shutil

# === 사용자 설정 ===
WEIGHTS = r"runs\detect\train4\weights\best.pt"  # 학습된 모델
IMAGES_DIR = r"C:\Users\User\Desktop\jo\2025 project\frames\bottle"   # 오토라벨링할 이미지 폴더
LABELS_DIR = r"C:\Users\User\Desktop\jo\2025 project\labels\bottle"   # 최종 라벨 저장 폴더(여기에만 남김)
IMG_SIZE = 720
CONF = 0.25
IOU = 0.6
BATCH = 16  
DEVICE = 0                # GPU:0, CPU는 'cpu'
OVERWRITE = False         # 같은 이름 .txt가 있을 때 덮어쓸지(False=스킵)

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts]

def main():
    images = list_images(IMAGES_DIR)
    if not images:
        print("[X] 이미지가 없습니다.")
        return

    # 1) '라벨 없는' 이미지만 선별
    labels_dir_path = Path(LABELS_DIR)
    unlabeled = []
    for img in images:
        if not (labels_dir_path / f"{img.stem}.txt").exists():
            unlabeled.append(img)

    print(f"[i] 전체 이미지: {len(images)} / 라벨 없는 이미지: {len(unlabeled)}")
    if not unlabeled:
        print("[✓] 신규 오토라벨링 대상이 없습니다.")
        return

    # 2) 모델 로드
    model = YOLO(WEIGHTS)

    # 3) 예측 실행 (라벨 저장: 5열만 저장되도록 save_conf=False)
    results = model.predict(
        source=[str(p) for p in unlabeled],
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        save_txt=True,
        save_conf=False,   # ★ 5열만 저장
        device=DEVICE,
        batch=BATCH,
        exist_ok=True,
        verbose=True
        # project/name 인자 제거 → 기본 runs/detect/predict* 에 임시 저장
    )
    if not results:
        print("[!] 예측 결과가 비어 있습니다.")
        return

    # YOLO가 실제로 저장한 임시 라벨 폴더 (기본: runs/detect/predict*/labels)
    save_dir = Path(results[0].save_dir)
    pred_labels_dir = save_dir / "labels"
    pred_txts = list(pred_labels_dir.glob("*.txt"))
    if not pred_txts:
        print("[!] 예측 라벨(.txt)이 생성되지 않았습니다. conf 값을 낮춰보세요.")
        return

    # 4) 최종 라벨 폴더로 이동(최종 폴더에만 존재하도록)
    labels_dir_path.mkdir(parents=True, exist_ok=True)
    copied, skipped, overwritten = 0, 0, 0

    for txt in pred_txts:
        dst = labels_dir_path / txt.name
        if dst.exists() and not OVERWRITE:
            skipped += 1
            continue
        if dst.exists() and OVERWRITE:
            overwritten += 1
        # 이번 예측은 save_conf=False로 이미 5열 → 그대로 이동
        shutil.move(str(txt), str(dst))
        copied += 1

    print(f"[✓] 라벨 저장 완료(5열): 신규 {copied}개, 스킵 {skipped}개, 덮어씀 {overwritten}개")
    print(f"[i] 라벨 최종 위치: {labels_dir_path}")

    # 5) 임시 YOLO 출력 폴더 정리
    shutil.rmtree(save_dir, ignore_errors=True)
    print(f"[i] 임시 폴더 정리: {save_dir}")

        # 4) 최종 라벨 폴더로 이동(최종 폴더에만 존재하도록)
    labels_dir_path.mkdir(parents=True, exist_ok=True)
    copied, skipped, overwritten = 0, 0, 0

    for txt in pred_txts:
        dst = labels_dir_path / txt.name
        if dst.exists() and not OVERWRITE:
            skipped += 1
            continue
        if dst.exists() and OVERWRITE:
            overwritten += 1
        shutil.move(str(txt), str(dst))
        copied += 1

    print(f"[✓] 라벨 저장 완료(5열): 신규 {copied}개, 스킵 {skipped}개, 덮어씀 {overwritten}개")
    print(f"[i] 라벨 최종 위치: {labels_dir_path}")

    # === 추가: 라벨 개수 확인 ===
    zero_labels, multi_labels = [], []
    for txt_file in labels_dir_path.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) == 0:
            zero_labels.append(txt_file.stem)   # 이미지 이름
        elif len(lines) > 1:
            multi_labels.append((txt_file.stem, len(lines)))

    if zero_labels:
        print(f"[!] 라벨 0개 (탐지 실패): {zero_labels}")
    if multi_labels:
        print(f"[!] 라벨 2개 이상 (다중 탐지): {multi_labels}")
    if not zero_labels and not multi_labels:
        print("[✓] 모든 라벨이 정상적으로 1개씩 생성됨.")


if __name__ == "__main__":
    main()
