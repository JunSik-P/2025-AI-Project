import os
import shutil
import random
from pathlib import Path
import yaml  # ← YAML 파싱

# === 사용자 설정 ===
ROOT_SRC = r"C:\Users\User\Desktop\jo\2025 project\labelling1400"   # 하위폴더들이 있는 루트
DATASET_DIR = r"C:\Users\User\Desktop\jo\2025 project\dataset_custom" # 최종 dataset 루트
USE_FOLDERS = ["magazine", "bottle"]  # ✅ 내가 사용할 하위폴더(클래스명)만 지정
VAL_RATIO = 0.2
SEED = 42
COPY = True  # True=복사, False=이동

# === 내부 유틸 ===
def _has_valid_label(txt_path: Path) -> bool:
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

def _load_names_from_yaml(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if isinstance(names, dict):
        # {0:'a',1:'b'} 형태 지원
        # 키가 문자열일 수도 있으니 정렬 시 int 변환 시도
        def _to_int(k):
            try:
                return int(k)
            except:
                return k
        names = [names[k] for k in sorted(names.keys(), key=_to_int)]
    assert isinstance(names, (list, tuple)), "[!] data.yaml의 names가 올바르지 않습니다."
    return list(names)

def relabel_from_yaml(yaml_path: Path, lbl_to_class_map):
    """
    lbl_to_class_map: [(dst_label_path: Path, class_name: str), ...]
    data.yaml의 names를 읽어 name->id 매핑을 만든 뒤,
    각 라벨(txt)의 첫 숫자(클래스 id)를 해당 class_name의 id로 교정.
    """
    names = _load_names_from_yaml(yaml_path)
    name_to_id = {str(n): i for i, n in enumerate(names)}
    # 대소문자/공백 대비 소문자 trim 매핑도 준비(유연하게)
    norm = lambda s: str(s).strip().lower()
    name_to_id_norm = {norm(k): v for k, v in name_to_id.items()}

    changed, skipped = 0, 0
    for txt_path, cname in lbl_to_class_map:
        target_id = name_to_id.get(cname)
        if target_id is None:
            target_id = name_to_id_norm.get(norm(cname))
        if target_id is None:
            print(f"[!] data.yaml에 '{cname}'가 없습니다. 라벨 교정 스킵 → {txt_path.name}")
            skipped += 1
            continue

        lines_out = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 해당 파일은 단일 클래스라 가정하고 모든 라인 0번 토큰을 target_id로 통일
                    parts[0] = str(target_id)
                    lines_out.append(" ".join(parts))
                else:
                    # 형식이 비정상이면 원문 보존
                    lines_out.append(line.rstrip("\n"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out))
        changed += 1

    print(f"[✓] 라벨 교정 완료: {changed}개 수정 (yaml에 없음으로 스킵 {skipped}개)")

def main():
    root_src = Path(ROOT_SRC)
    dataset_root = Path(DATASET_DIR)

    # ✅ 클래스 이름은 사용자가 지정한 USE_FOLDERS 로 결정
    class_names = USE_FOLDERS
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

    # data.yaml 작성
    write_data_yaml(dataset_root, class_names)

    # data.yaml의 names 기준으로 라벨 파일의 클래스 id 일괄 교정
    relabel_from_yaml(dataset_root / "data.yaml", train_map + val_map)

    # 개수 출력
    n_train_img = len([*train_img.glob("*")])
    n_train_lbl = len([*train_lbl.glob("*.txt")])
    n_val_img   = len([*val_img.glob("*")])
    n_val_lbl   = len([*val_lbl.glob("*.txt")])
    print(f"[i] 개수 확인 → train(images/labels)={n_train_img}/{n_train_lbl}, "
          f"val(images/labels)={n_val_img}/{n_val_lbl}")

    print("\n학습 실행 예:")
    print(f'yolo detect train model=yolo11n.pt data="{(dataset_root / "data.yaml")}" imgsz=640 '
          f'epochs=200 batch=16 seed=42 patience=30')

if __name__ == "__main__":
    main()
