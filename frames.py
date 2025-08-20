import cv2
import os
from pathlib import Path

def extract_frames_fixed_count(video_path, output_dir, prefix, target_frames):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[X] {video_path} 열기 실패")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(1, total_frames // target_frames)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= target_frames:
            break
        if frame_idx % interval == 0:
            filename = f"{prefix}_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[✓] {prefix}: {saved} frames saved to {output_dir}")


# === 영상 바로 들어있는 경우 ===
videos_root = r"C:\Users\User\Desktop\jo\2025 project\videos\bottle"
frames_root = r"C:\Users\User\Desktop\jo\2025 project\frames\bottle"
target_class_frame_count = 1000  # 전체 합쳐서 뽑을 장수

video_files = [f for f in os.listdir(videos_root) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
n_videos = len(video_files)
if n_videos == 0:
    print("[X] 영상 파일이 없습니다.")
else:
    per_video_frame = max(1, target_class_frame_count // n_videos)
    os.makedirs(frames_root, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(videos_root, video_file)
        prefix = os.path.splitext(video_file)[0]
        extract_frames_fixed_count(
            video_path,
            frames_root,
            prefix,
            target_frames=per_video_frame
        )
