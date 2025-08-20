import os
import shutil
import random
from pathlib import Path
import yaml
from ultralytics import YOLO

# === 사용자 설정 ===
IMAGES_DIR = Path(r"C:\Users\User\Desktop\images_all")     # 전체 이미지
SAMPLE_DIR = Path(r"C:\Users\User\Desktop\sample_labels")  # 샘플 라벨링 (img+txt)
DATASET_DIR = Path(r"C:\Users\User\Desktop\dataset_tmp")   # 임시 학습용 dataset
VAL_RATIO = 0.2
SEED = 42
EPOCHS = 50
BATCH = 16
IMG_SIZE = 640

# === 내부 유틸 ===
def collect_pairs(img_dir: Path, lbl_dir: Path):
    img_exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
    pairs = []
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs

def split_pairs(pairs, val_ratio=0.2, seed=42):
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n_val = max(1, int(len(pairs)*val_ratio))
    return pairs[n_val:], pairs[:n_val]

def copy_pairs(pairs, img_dst: Path, lbl_dst: Path):
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_dst / img.name)
        shutil.copy2(lbl, lbl_dst / lbl.name)

def write_data_yaml(dataset_root: Path, class_names):
    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    return yaml_path

# === 메인 파이프라인 ===
def main():
    # 1) 샘플 라벨링 데이터셋 준비
    pairs = collect_pairs(SAMPLE_DIR, SAMPLE_DIR)
    train_pairs, val_pairs = split_pairs(pairs, VAL_RATIO, SEED)

    train_img = DATASET_DIR/"images"/"train"
    train_lbl = DATASET_DIR/"labels"/"train"
    val_img   = DATASET_DIR/"images"/"val"
    val_lbl   = DATASET_DIR/"labels"/"val"

    copy_pairs(train_pairs, train_img, train_lbl)
    copy_pairs(val_pairs,   val_img,   val_lbl)

    # 클래스 이름은 샘플 라벨 txt에서 추출할 수 있으나,
    # 여기서는 폴더명이 클래스명이라고 가정
    CLASS_NAMES = [SAMPLE_DIR.name]
    data_yaml = write_data_yaml(DATASET_DIR, CLASS_NAMES)

    # 2) YOLO 학습
    model = YOLO("yolov8n.pt")  # 가벼운 사전학습 모델로 시작
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        seed=SEED
    )
    best_weight = Path(model.trainer.best)

    # 3) 샘플에 없는 나머지 이미지 자동 라벨링
    all_imgs = {p.name: p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]}
    labeled_imgs = {p.name for p, _ in pairs}
    unlabeled = [all_imgs[n] for n in all_imgs if n not in labeled_imgs]

    if unlabeled:
        print(f"[i] 자동 라벨링 시작 ({len(unlabeled)}개)")
        pred_dir = SAMPLE_DIR/"pred_tmp"
        model = YOLO(best_weight)
        model.predict(
            source=[str(p) for p in unlabeled],
            conf=0.25,
            iou=0.6,
            imgsz=IMG_SIZE,
            save_txt=True,
            save_conf=False,
            project=str(pred_dir),
            name="labels",
            exist_ok=True
        )
        # 4) 생성된 라벨 이동 → SAMPLE_DIR
        gen_lbl_dir = next((pred_dir/"labels").glob("*"), None)
        if gen_lbl_dir and gen_lbl_dir.exists():
            for txt in gen_lbl_dir.glob("*.txt"):
                shutil.move(str(txt), SAMPLE_DIR/txt.name)
        shutil.rmtree(pred_dir, ignore_errors=True)

    print("[✓] 최종 라벨링 완료. SAMPLE_DIR 안에 전체 라벨 확보.")

if __name__ == "__main__":
    main()
