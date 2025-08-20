import os
import shutil
import random
from pathlib import Path

# === 사용자 설정 ===
IMAGES_SRC = r"C:\Users\User\Desktop\jo\2025 project\mini_labels\bottle"   # 이미지가 있는 폴더(라벨과 같은 폴더여도 됨)
LABELS_SRC = r"C:\Users\User\Desktop\jo\2025 project\mini_labels\bottle"   # 라벨(.txt) 폴더
DATASET_DIR = r"C:\Users\User\Desktop\jo\2025 project\dataset\bottle"      # 최종 dataset 루트
CLASS_NAMES = ["bottle"]  # YOLO 클래스 이름 목록
VAL_RATIO = 0.2
SEED = 42
COPY = True  # True면 복사, False면 이동 (원본 보존/이동)

# === 내부 유틸 ===
def _has_valid_label(txt_path: Path) -> bool:
    """YOLO 포맷 라벨이 비어있지 않고 숫자 5개 이상(클래스 cx cy w h …)인지 대충 확인."""
    if not txt_path.exists() or txt_path.suffix.lower() != ".txt":
        return False
    try:
        content = txt_path.read_text(encoding="utf-8").strip()
        if not content:
            return False
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                return False
            # 숫자 형식 대략 체크
            _ = [float(x) for x in parts[0:5]]
        return True
    except Exception:
        return False

def collect_pairs(images_dir: Path, labels_dir: Path):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    images.sort(key=lambda p: p.name.lower())

    pairs = []
    missing = []
    badlabel = []

    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            if _has_valid_label(lbl):
                pairs.append((img, lbl))
            else:
                badlabel.append(lbl)
        else:
            missing.append(img)

    return pairs, missing, badlabel

def split_pairs(pairs, val_ratio=0.2, seed=42):
    assert 0.0 <= val_ratio < 1.0, "VAL_RATIO는 0 이상 1 미만이어야 합니다."
    rnd = random.Random(seed)
    pairs = pairs[:]  # copy
    rnd.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    n_val = min(max(n_val, 1 if len(pairs) > 0 else 0), len(pairs))  # 최소 1장, 최대 len
    return pairs[n_val:], pairs[:n_val]  # train, val

def _unique_dst(dst_path: Path) -> Path:
    """동일 파일명이 이미 존재하면 _1, _2 …를 붙여 충돌 방지."""
    if not dst_path.exists():
        return dst_path
    stem, suf = dst_path.stem, dst_path.suffix
    k = 1
    while True:
        cand = dst_path.with_name(f"{stem}_{k}{suf}")
        if not cand.exists():
            return cand
        k += 1

def copy_or_move(pairs, img_dst: Path, lbl_dst: Path, do_copy=True):
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        img_to = _unique_dst(img_dst / img.name)
        lbl_to = _unique_dst(lbl_dst / (img_to.stem + ".txt"))  # 이름 충돌 시 라벨도 매칭되게
        if do_copy:
            shutil.copy2(img, img_to)
            shutil.copy2(lbl, lbl_to)
        else:
            shutil.move(str(img), str(img_to))
            shutil.move(str(lbl), str(lbl_to))

def write_data_yaml(dataset_root: Path, class_names):
    yaml_path = dataset_root / "data.yaml"
    path_line = dataset_root.resolve().as_posix()
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {path_line}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    print(f"[✓] data.yaml 생성 → {yaml_path}")

def main():
    images_dir = Path(IMAGES_SRC)
    labels_dir = Path(LABELS_SRC)
    root = Path(DATASET_DIR)

    print(f"[i] IMAGES_SRC = {images_dir}")
    print(f"[i] LABELS_SRC = {labels_dir}")
    print(f"[i] DATASET_DIR = {root}")

    pairs, missing, badlabel = collect_pairs(images_dir, labels_dir)

    if missing:
        print(f"[!] 라벨 누락 이미지 {len(missing)}개 (예시 5개):")
        for p in missing[:5]:
            print("   -", p.name)
    if badlabel:
        print(f"[!] 깨진/빈 라벨 {len(badlabel)}개 (예시 5개):")
        for p in badlabel[:5]:
            print("   -", p.name)

    if not pairs:
        print("[X] 매칭된 (이미지, 라벨) 쌍이 없습니다. 경로/파일/라벨 포맷을 확인하세요.")
        return

    train_pairs, val_pairs = split_pairs(pairs, VAL_RATIO, SEED)
    print(f"[i] 매칭 쌍: {len(pairs)} → train={len(train_pairs)}, val={len(val_pairs)} (val_ratio={VAL_RATIO})")

    # 폴더 준비
    train_img = root / "images" / "train"
    train_lbl = root / "labels" / "train"
    val_img   = root / "images" / "val"
    val_lbl   = root / "labels" / "val"

    copy_or_move(train_pairs, train_img, train_lbl, do_copy=COPY)
    copy_or_move(val_pairs,   val_img,   val_lbl,   do_copy=COPY)
    print("[✓] 파일 배치 완료.")

    write_data_yaml(root, CLASS_NAMES)

    # 빠른 개수 확인
    n_train_img = len([*train_img.glob("*")])
    n_train_lbl = len([*train_lbl.glob("*.txt")])
    n_val_img   = len([*val_img.glob("*")])
    n_val_lbl   = len([*val_lbl.glob("*.txt")])
    print(f"[i] 개수 확인 → train(images/labels)={n_train_img}/{n_train_lbl}, "
          f"val(images/labels)={n_val_img}/{n_val_lbl}")

    print("\n학습 실행 예:")
    print(f'yolo detect train model=yolo11n.pt data="{(root / "data.yaml")}" imgsz=1280 '
          f'epochs=200 batch=16 seed=42 patience=80 freeze=10')

if __name__ == "__main__":
    main()
