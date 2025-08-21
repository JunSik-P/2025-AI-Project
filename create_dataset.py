import os
import shutil
import random
from pathlib import Path
import yaml  # (지금은 읽기엔 사용하지 않지만, 추후 확장 대비 해서 유지)

# === 사용자 설정 ===
ROOT_SRC = r"C:\Users\User\Desktop\jo\PROJECT\labelling1400"   # 하위폴더들이 있는 루트
DATASET_DIR = r"C:\Users\User\Desktop\jo\PROJECT\dataset"      # 최종 dataset 루트
USE_FOLDERS = ["magazine", "bottle", "boots_black", "boots_brown", "bulletproof_plate",
               "canteen", "gas_mask", "gas_mask_pouch", "helmet", "MRE"]  # 사용할 클래스(폴더)들
VAL_RATIO = 0.2
SEED = 42
COPY = True  # True=복사, False=이동

# === 내부 유틸 ===
def _has_valid_label(txt_path: Path) -> bool:
    if not txt_path.exists() or txt_path.suffix.lower() != ".txt":
        return False
    try:
        # BOM이 있더라도 안전하게 읽기
        content = txt_path.read_text(encoding="utf-8-sig").strip()
        if not content:
            return False
        for line in content.splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                # 빈 줄/주석 라인은 무시
                continue
            parts = raw.split()
            if len(parts) < 5:
                return False
            # 앞 5개 토큰이 숫자로 파싱되는지만 확인 (class cx cy w h)
            _ = [float(x) for x in parts[0:5]]
        return True
    except Exception:
        return False

def collect_pairs(images_dir: Path, labels_dir: Path):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    images.sort(key=lambda p: p.name.lower())

    pairs, missing, badlabel = [], [], []
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
    rnd = random.Random(seed)
    pairs = pairs[:]
    rnd.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    n_val = min(max(n_val, 1 if len(pairs) > 0 else 0), len(pairs))
    return pairs[n_val:], pairs[:n_val]

def _unique_dst(dst_path: Path) -> Path:
    if not dst_path.exists():
        return dst_path
    stem, suf = dst_path.stem, dst_path.suffix
    k = 1
    while True:
        cand = dst_path.with_name(f"{stem}_{k}{suf}")
        if not cand.exists():
            return cand
        k += 1

def copy_or_move(pairs_with_class, img_dst: Path, lbl_dst: Path, do_copy=True):
    """
    pairs_with_class: [(img_path, lbl_path, class_name), ...]
    반환: [(dst_label_path, class_name), ...]  ← 라벨 교정용 매핑
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    mapping = []
    for img, lbl, cname in pairs_with_class:
        img_to = _unique_dst(img_dst / img.name)
        # 라벨 파일명은 항상 이미지 stem에 맞추어 생성
        lbl_to = _unique_dst(lbl_dst / (img_to.stem + ".txt"))
        if do_copy:
            shutil.copy2(img, img_to)
            shutil.copy2(lbl, lbl_to)
        else:
            shutil.move(str(img), str(img_to))
            shutil.move(str(lbl), str(lbl_to))
        mapping.append((lbl_to, cname))
    return mapping

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

# === 핵심: YAML을 다시 읽지 않고 즉석 매핑으로 교정 ===
def relabel_with_map(lbl_to_class_map, class_to_id: dict):
    """
    lbl_to_class_map: [(dst_label_path: Path, class_name: str), ...]
    class_to_id: {'magazine': 0, 'bottle': 1, ...}  # USE_FOLDERS 순서 그대로
    """
    def norm(s: str) -> str:
        # 대소문자/공백/하이픈/언더스코어 차이를 흡수
        return str(s).strip().lower().replace(" ", "").replace("-", "_")

    # 원문 키 + 정규화 키 모두 지원
    class_to_id_norm = {norm(k): v for k, v in class_to_id.items()}

    changed, skipped = 0, 0
    for txt_path, cname in lbl_to_class_map:
        cid = class_to_id.get(cname)
        if cid is None:
            cid = class_to_id_norm.get(norm(cname))

        if cid is None:
            print(f"[!] 클래스 매핑에 '{cname}' 없음 → 스킵: {txt_path}")
            skipped += 1
            continue

        if not txt_path.exists():
            print(f"[!] 라벨 파일 없음 → 스킵: {txt_path}")
            skipped += 1
            continue

        lines_out = []
        with open(txt_path, "r", encoding="utf-8-sig") as f:  # BOM 안전
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    lines_out.append(line.rstrip("\n"))
                    continue
                parts = raw.split()
                if len(parts) >= 5:
                    parts[0] = str(cid)
                    lines_out.append(" ".join(parts))
                else:
                    # 형식이 비정상인 줄은 원문 보존
                    lines_out.append(line.rstrip("\n"))

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out))
        changed += 1

    print(f"[✓] 라벨 교정 완료: {changed}개 수정 (스킵 {skipped}개)")

def _count_files(dir_path: Path, exts=None):
    if exts is None:
        return len([p for p in dir_path.iterdir() if p.is_file()])
    exts = {e.lower() for e in exts}
    return len([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])

def main():
    root_src = Path(ROOT_SRC)
    dataset_root = Path(DATASET_DIR)

    # ✅ 클래스 이름은 사용자가 지정한 USE_FOLDERS 로 결정
    class_names = list(USE_FOLDERS)  # 순서 고정
    print(f"[i] 사용할 클래스: {class_names}")

    all_pairs_with_class = []
    all_missing, all_bad = [], []

    for cname in class_names:
        cdir = root_src / cname
        if not cdir.exists():
            print(f"[!] {cdir} 폴더 없음 → 건너뜀")
            continue
        pairs, missing, badlabel = collect_pairs(cdir, cdir)
        print(f"[{cname}] 매칭 {len(pairs)}, 누락 {len(missing)}, 깨짐 {len(badlabel)}")
        # (img, lbl, class_name) 형태로 확장
        all_pairs_with_class.extend([(img, lbl, cname) for (img, lbl) in pairs])
        all_missing.extend(missing)
        all_bad.extend(badlabel)

    if not all_pairs_with_class:
        print("[X] 선택한 폴더들에서 매칭된 (이미지, 라벨) 쌍이 없습니다.")
        return

    train_pairs, val_pairs = split_pairs(all_pairs_with_class, VAL_RATIO, SEED)
    print(f"[i] 전체 매칭 쌍 {len(all_pairs_with_class)} → train={len(train_pairs)}, val={len(val_pairs)}")

    train_img = dataset_root / "images" / "train"
    train_lbl = dataset_root / "labels" / "train"
    val_img   = dataset_root / "images" / "val"
    val_lbl   = dataset_root / "labels" / "val"

    # 복사/이동 + (목적지 라벨 경로, 클래스명) 매핑 회수
    train_map = copy_or_move(train_pairs, train_img, train_lbl, do_copy=COPY)
    val_map   = copy_or_move(val_pairs,   val_img,   val_lbl,   do_copy=COPY)
    print("[✓] 파일 배치 완료.")

    # data.yaml 작성 (학습 편의용)
    write_data_yaml(dataset_root, class_names)

    # 🔁 USE_FOLDERS 순서로 class→id 매핑을 만들고, 모든 라벨의 클래스 인덱스를 교정
    class_to_id = {name: i for i, name in enumerate(class_names)}
    relabel_with_map(train_map + val_map, class_to_id)

    # 개수 출력
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    n_train_img = _count_files(train_img, img_exts)
    n_train_lbl = _count_files(train_lbl, {".txt"})
    n_val_img   = _count_files(val_img,   img_exts)
    n_val_lbl   = _count_files(val_lbl,   {".txt"})
    print(f"[i] 개수 확인 → train(images/labels)={n_train_img}/{n_train_lbl}, "
          f"val(images/labels)={n_val_img}/{n_val_lbl}")

    # 샘플 3개만 확인 출력
    sample = (train_map + val_map)[:3]
    for p, c in sample:
        if p.exists():
            print(f"\n=== {p.name} (class='{c}') 앞부분 미리보기 ===")
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    for i, line in enumerate(f):
                        if i >= 3: break
                        print(line.strip())
            except Exception as e:
                print(f"(미리보기 실패: {e})")

    print("\n학습 실행 예:")
    print(f'yolo detect train model=yolo11n.pt data="{(dataset_root / "data.yaml")}" '
          f'imgsz=640 epochs=200 batch=16 seed=42 patience=30')

if __name__ == "__main__":
    main()

