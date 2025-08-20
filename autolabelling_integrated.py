import os
import shutil
import glob
import random
import yaml
from ultralytics import YOLO  # YOLOv8 기준 (YOLOv5도 사용 가능)

# ============ 사용자 입력 ============
PROJECT_DIR = "project"
TARGET_CLASS = "bottle"   # 🔥 여기만 바꿔서 원하는 클래스 지정
IMG_EXTS = [".jpg", ".png", ".jpeg"]
# ===================================

# 경로 설정
images_path = os.path.join(PROJECT_DIR, "images", TARGET_CLASS)
labels_path = os.path.join(PROJECT_DIR, "labels", TARGET_CLASS)
dataset_tmp_root = os.path.join(PROJECT_DIR, "dataset_tmp")
dataset_tmp_class = os.path.join(dataset_tmp_root, f"dataset_tmp_{TARGET_CLASS}")

# dataset_tmp 폴더 초기화
if os.path.exists(dataset_tmp_class):
    shutil.rmtree(dataset_tmp_class)
os.makedirs(os.path.join(dataset_tmp_class, "images/train"), exist_ok=True)
os.makedirs(os.path.join(dataset_tmp_class, "images/val"), exist_ok=True)
os.makedirs(os.path.join(dataset_tmp_class, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(dataset_tmp_class, "labels/val"), exist_ok=True)

# 1. train/val 분할
all_images = [f for f in glob.glob(os.path.join(images_path, "*")) if os.path.splitext(f)[1].lower() in IMG_EXTS]
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_images, val_images = all_images[:split_idx], all_images[split_idx:]

def copy_with_label(img_list, split):
    for img_file in img_list:
        base = os.path.basename(img_file)
        name, _ = os.path.splitext(base)
        label_file = os.path.join(labels_path, f"{name}.txt")

        shutil.copy(img_file, os.path.join(dataset_tmp_class, f"images/{split}", base))
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(dataset_tmp_class, f"labels/{split}", f"{name}.txt"))

copy_with_label(train_images, "train")
copy_with_label(val_images, "val")

# 2. data.yaml 생성
data_yaml = {
    "train": os.path.join(dataset_tmp_class, "images/train"),
    "val": os.path.join(dataset_tmp_class, "images/val"),
    "nc": 1,
    "names": [TARGET_CLASS]
}
with open(os.path.join(dataset_tmp_class, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print(f"[INFO] YOLO dataset created at {dataset_tmp_class}")

# 3. YOLO 학습
model = YOLO("yolov8n.pt")  # YOLOv8n 사전학습 모델 사용 (작고 빠름)
model.train(data=os.path.join(dataset_tmp_class, "data.yaml"), epochs=50, imgsz=640)

# 4. 자동 라벨링 수행 (샘플 라벨 제외한 나머지)
results = model.predict(source=images_path, save=False)

# 5. 라벨 저장
for r in results:
    img_name = os.path.basename(r.path)
    name, _ = os.path.splitext(img_name)
    label_file = os.path.join(labels_path, f"{name}.txt")

    with open(label_file, "w") as f:
        for box in r.boxes.xywhn.tolist():  # xywh normalized
            cls = 0  # 단일 클래스
            x, y, w, h = box
            f.write(f"{cls} {x} {y} {w} {h}\n")

print(f"[INFO] Labels saved at {labels_path}")

# 6. 검증: 라벨 박스 개수 체크
problem_cases = []
for txt_file in glob.glob(os.path.join(labels_path, "*.txt")):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    if len(lines) != 1:
        problem_cases.append((os.path.basename(txt_file), len(lines)))

if problem_cases:
    print("\n[WARNING] Some images have abnormal label counts:")
    for fname, cnt in problem_cases:
        print(f"  - {fname}: {cnt} boxes")
else:
    print("\n[INFO] All images have exactly 1 label box ✅")
