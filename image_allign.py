from pathlib import Path
import shutil

# ==== 사용자 설정 ====
PROJECT_DIR = Path(r"C:\Users\user\Desktop\project")   # project 폴더 경로
CLASS = "bulletproof_plate"                  # 작업할 클래스
COPY = True                                  # True=복사, False=이동
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# ==== 경로 ====
images_dir = PROJECT_DIR / "images" / CLASS
labels_dir = PROJECT_DIR / "labels" / CLASS
labels_dir.mkdir(parents=True, exist_ok=True)

# ==== 실행 ====
txts = sorted([p for p in labels_dir.glob("*.txt")])
if not txts:
    print(f"[!] {labels_dir} 안에 샘플 라벨(txt)이 없습니다.")
    raise SystemExit

copied, missing = 0, []
for txt in txts:
    stem = txt.stem  # 예: bulletproof_plate_00010
    # images 폴더에서 같은 stem의 이미지를 확장자별로 검색
    found_img = None
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            found_img = cand
            break

    if found_img is None:
        missing.append(stem)
        continue

    dst = labels_dir / found_img.name
    if dst.exists():
        # 이미 labels 폴더에 이미지가 들어있으면 스킵
        continue

    if COPY:
        shutil.copy2(found_img, dst)
    else:
        shutil.move(str(found_img), str(dst))
    copied += 1

print(f"[✓] 작업 완료: {copied}개 이미지 {'복사' if COPY else '이동'}됨.")
if missing:
    print(f"[!] 매칭 이미지 누락 {len(missing)}개 (images/{CLASS}에서 못 찾음):")
    # 너무 길어지지 않게 앞 몇 개만 미리보기
    preview = ", ".join(missing[:10])
    print(f"    예시: {preview}{' ...' if len(missing) > 10 else ''}")
