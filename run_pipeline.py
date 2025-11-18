# run_pipeline.py
import sys, os, json, glob, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PY = sys.executable

def pick_latest_summary(out_root: Path, out_name: str, base_stem: str) -> Path:
    """
    out_root/out_name*/**/<base>_summary.json 중 '가장 최근' 파일을 선택.
    (_1, _2 같은 접미사가 붙어도 안전)
    """
    pattern = os.path.join(str(out_root), f"{out_name}*", f"{base_stem}_summary.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return Path(candidates[0])

def main():
    print("분석할 영상의 전체 경로를 입력하세요:")
    video_in = input("VIDEO: ").strip().strip('"')
    if not video_in:
        print("❌ 영상 경로가 비었습니다."); sys.exit(2)

    video_path = Path(video_in)
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}"); sys.exit(2)

    out_root = ROOT / "output"
    out_root.mkdir(parents=True, exist_ok=True)

    # 폴더 이름은 시각 기반. (여기서 폴더를 미리 만들지 않음! 충돌 회피 로직은 run_infer가 담당)
    out_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # 1) 추론 실행
    infer_cmd = [
        PY, str(ROOT / "run_infer.py"),
        "--video", str(video_path),
        "--out-root", str(out_root),
        "--out-name", out_name,
    ]
    print("[pipeline] run infer:", " ".join(infer_cmd))
    subprocess.check_call(infer_cmd)

    # 2) summary.json 파싱 (실제 생성된 폴더를 탐색)
    base = video_path.stem
    summary_path = pick_latest_summary(out_root, out_name, base)
    if not summary_path or not summary_path.exists():
        print("❌ summary.json을 찾지 못했습니다.")
        print(f"  검색 패턴: {out_root}\\{out_name}*\\{base}_summary.json")
        sys.exit(3)

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"❌ summary.json 읽기 실패: {summary_path}\n{e}")
        sys.exit(3)

    overlay_mp4 = Path(summary["outputs"]["overlay_video"])
    result_jsonl = Path(summary["outputs"]["frame_log_jsonl"])

    if not overlay_mp4.exists() or not result_jsonl.exists():
        print("❌ 산출물(mp4/jsonl) 확인 실패.")
        print(" overlay:", overlay_mp4)
        print(" jsonl  :", result_jsonl)
        print(" summary:", summary_path)
        sys.exit(4)

    # 3) GUI 실행 (분석된 mp4 재생)
    gui_cmd = [
        PY, str(ROOT / "run_gui.py"),
        "--video", str(overlay_mp4),
        "--result-jsonl", str(result_jsonl),
    ]
    print("[pipeline] run gui:", " ".join(gui_cmd))
    subprocess.check_call(gui_cmd)

if __name__ == "__main__":
    main()
