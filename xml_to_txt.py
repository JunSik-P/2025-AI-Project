# voc2yolo_singleclass_bottle_cleanup.py
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

# 변환할 폴더 경로 (필요시 수정)
LABEL_DIR = Path(r"C:\Users\user\Desktop\project\labels\bottle")
# 단일 클래스 학습이라 class_id=0 (필요하면 변경)
CLASS_ID = 0

def convert_bbox(size, box):
    """VOC -> YOLO 좌표 변환"""
    w, h = size
    xmin, ymin, xmax, ymax = box
    x = ((xmin + xmax) / 2.0) / w
    y = ((ymin + ymax) / 2.0) / h
    bw = (xmax - xmin) / float(w)
    bh = (ymax - ymin) / float(h)
    return (x, y, bw, bh)

def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"'size' 태그 없음: {xml_path.name}")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))

    objs = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))

        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w - 1))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h - 1))

        if xmax <= xmin or ymax <= ymin:
            continue
        objs.append((w, h, (xmin, ymin, xmax, ymax)))
    return objs

def main():
    xml_files = sorted(glob.glob(str(LABEL_DIR / "*.xml")))
    if not xml_files:
        print(f"[알림] XML 파일이 없습니다: {LABEL_DIR}")
        return

    converted, empty, deleted = 0, 0, 0
    for x in xml_files:
        xml_path = Path(x)
        stem = xml_path.stem
        yolo_txt = xml_path.with_suffix(".txt")

        try:
            objs = parse_xml(xml_path)
        except Exception as e:
            print(f"[스킵] 파싱 실패: {xml_path.name} -> {e}")
            continue

        lines = []
        for (W, H, (xmin, ymin, xmax, ymax)) in objs:
            x, y, bw, bh = convert_bbox((W, H), (xmin, ymin, xmax, ymax))
            lines.append(f"{CLASS_ID} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

        # 객체가 없으면 빈 txt라도 생성
        if not lines:
            empty += 1

        with open(yolo_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        converted += 1

        # 변환 성공 시 XML 삭제
        try:
            xml_path.unlink()
            deleted += 1
        except Exception as e:
            print(f"[경고] XML 삭제 실패: {xml_path.name} -> {e}")

    print(f"\n✅ 변환 완료: {converted}개 txt 생성")
    if empty:
        print(f"ℹ️ 객체 0개(빈 txt): {empty}개")
    print(f"🗑️ XML 삭제 완료: {deleted}개")

if __name__ == "__main__":
    main()

