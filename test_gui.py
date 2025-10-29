# -*- coding: utf-8 -*-
"""
StorageMonitorUI – 1.5초 동시 인식 트리거 (필드명 수정 반영 / 경로 직접 지정)

- 좌측: 단일 mp4 재생(세로 영상 자동 회전)
- 우측: 시간/이름/반납·불출/품목 로그 테이블
- *_result.jsonl 을 프레임 스트림처럼 따라가며, 다음 조건을 1.5초 연속 충족 시 모달 표시:
    얼굴(person_id != 'Unknown' & similarity >= 0.40) AND 아이템(class_name != 'Unknown')
- 사용자가 [불출/반납/건너뛰기] 선택 → *_decisions.jsonl 에 기록, 로그 테이블 반영
- 재무장 규칙: 이후 0.5초 이상 비충족되기 전까지 재트리거 금지

필수 패키지: PyQt5, opencv-python, numpy
실행 예시: python storage_monitor_ui_sync_1p5s_fixed.py
"""
from __future__ import annotations
import os, sys, json, logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QSize, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QDialog, QDialogButtonBox,
    QFormLayout, QMessageBox, QSizePolicy
)

# ======================
# 경로 (직접 입력)
# ======================
TEST_VIDEO_PATH   = r"C:\\Users\\User\\Desktop\\csv\\output\\park_clean_ver\\park_clean_merged_result.mp4"
TEST_RESULT_JSONL = r"C:\\Users\\User\\Desktop\\csv\\output\\park_clean_ver\\park_clean_merged_result.jsonl"

# decisions 파일은 result.jsonl 프리픽스를 따라 같은 폴더에 생성됩니다
RESULT_PREFIX = Path(TEST_RESULT_JSONL).name.replace("_result.jsonl", "")
TEST_DECISIONS_JSONL = str(Path(TEST_RESULT_JSONL).with_name(f"{RESULT_PREFIX}_decisions.jsonl"))

# 파라미터 (필요시 조정)
HOLD_MIN_S = 1.5    # 동시 인식 최소 지속 시간
RESET_MIN_S = 0.5   # 비충족 연속 지속(재무장) 시간
FACE_SIM_THR = 0.40 # 얼굴 최소 유사도

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("UI1p5")

# ======================
# 비디오 플레이어
# ======================
class VideoPlayer(QWidget):
    def __init__(self, video_path: str, aspect_ratio=(9,16)):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"영상 열기 실패: {video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_s = self.nframes / self.fps if self.fps > 0 else 0.0
        self.aspect_ratio = aspect_ratio

        self.label = QLabel("영상 로딩 중…")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.playing = True
        self.frame_idx = 0
        self.timer.start(max(1, int(1000/(self.fps or 30))))

    def sizeHint(self):
        return QSize(540, 960)

    def time_s(self) -> float:
        return self.frame_idx / self.fps if self.fps > 0 else 0.0

    def pause(self):
        if self.playing:
            self.playing = False
            self.timer.stop()

    def play(self):
        if not self.playing:
            self.playing = True
            self.timer.start(max(1, int(1000/(self.fps or 30))))

    def _on_tick(self):
        if not self.playing:
            return
        if self.frame_idx >= self.nframes:
            self.pause(); return
        self.frame_idx += 1
        self._read_and_show()

    def _read_and_show(self):
        ok, frame = self.cap.read()
        if not ok:
            self.pause(); return
        # 세로 보정(원본이 가로면 90° 회전)
        if self.aspect_ratio == (9,16) and frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w*ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.label.setPixmap(pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

# ======================
# 토스트 배너(알림)
# ======================
class ToastBanner(QLabel):
    def __init__(self):
        super().__init__("")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: rgba(0,0,0,0.72); color:#fff; padding:8px 12px; border-radius:10px;")
        self.hide()
        self.timer = QTimer(self); self.timer.timeout.connect(self.hide)
    def show_text(self, text: str, msec=2000):
        self.setText(text); self.show(); self.raise_(); self.timer.start(msec)

# ======================
# Result JSONL 로더/질의
# ======================
@dataclass
class FrameRec:
    t_s: float
    faces: List[dict]
    items: List[dict]

class ResultStream:
    def __init__(self, path: str):
        self.frames: List[FrameRec] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t_ms = o.get('timestamp_ms')
                if t_ms is None:
                    continue
                faces = o.get('faces', []) or []
                items = o.get('items', []) or []
                self.frames.append(FrameRec(t_s=float(t_ms)/1000.0, faces=faces, items=items))
        self.frames.sort(key=lambda r: r.t_s)
        self.idx = 0

    def reset_to_time(self, t: float):
        lo, hi = 0, len(self.frames)
        while lo < hi:
            mid = (lo+hi)//2
            if self.frames[mid].t_s < t:
                lo = mid+1
            else:
                hi = mid
        self.idx = max(0, lo-1)

    def get_state_at(self, t: float) -> Tuple[bool, Optional[str], Optional[str]]:
        """시각 t에서 동시 인식 여부, 대표 얼굴/품목을 리턴.
        - 얼굴 대표: person_id != 'Unknown' & similarity>=FACE_SIM_THR 인 후보 중 **최고 similarity**
        - 품목 대표: class_name != 'Unknown' 첫 후보
        """
        while self.idx+1 < len(self.frames) and self.frames[self.idx+1].t_s <= t:
            self.idx += 1
        if not self.frames:
            return False, None, None
        fr = self.frames[self.idx]

        face_name: Optional[str] = None
        best = None
        for f in fr.faces:
            name = str(f.get('person_id', '') or '').strip()
            sim  = float(f.get('similarity', 0.0))
            if name and name.lower() != 'unknown' and sim >= FACE_SIM_THR:
                if best is None or sim > best[1]:
                    best = (name, sim)
        if best:
            face_name = best[0]

        item_class: Optional[str] = None
        for it in fr.items:
            cls = str(it.get('class_name', '') or '').strip()
            if cls and cls.lower() != 'unknown':
                item_class = cls
                break

        ok = (face_name is not None) and (item_class is not None)
        return ok, face_name, item_class

# ======================
# 결정 모달
# ======================
class DecisionDialog(QDialog):
    def __init__(self, person: str, item: str, t_s: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이벤트 분류")
        self.choice = None
        form = QFormLayout()
        form.addRow("시간(초)", QLabel(f"{t_s:.2f}"))
        form.addRow("이름", QLabel(person or "(미확정)"))
        form.addRow("품목", QLabel(item or "(미확정)"))
        btns = QDialogButtonBox()
        b_issue = QPushButton("불출"); b_return = QPushButton("반납"); b_skip = QPushButton("건너뛰기")
        btns.addButton(b_issue, QDialogButtonBox.AcceptRole)
        btns.addButton(b_return, QDialogButtonBox.AcceptRole)
        btns.addButton(b_skip, QDialogButtonBox.RejectRole)
        b_issue.clicked.connect(lambda: self._set_choice("불출"))
        b_return.clicked.connect(lambda: self._set_choice("반납"))
        b_skip.clicked.connect(self.reject)
        lay = QVBoxLayout(self); lay.addLayout(form); lay.addWidget(btns)
    def _set_choice(self, c):
        self.choice = c; self.accept()

# ======================
# 메인 윈도우
# ======================
class MainWindow(QMainWindow):
    def __init__(self, video_path: str, result_jsonl: str, decisions_out: str):
        super().__init__()
        self.setWindowTitle("StorageMonitorUI – 1.5초 동시 인식")
        self.resize(1280, 820)

        # 좌: 비디오, 우: 로그
        self.player = VideoPlayer(video_path, aspect_ratio=(9,16))
        self.stream = ResultStream(result_jsonl)
        self.decisions_out = decisions_out
        Path(self.decisions_out).parent.mkdir(parents=True, exist_ok=True)

        self.banner = ToastBanner()
        left = QVBoxLayout(); c = QWidget(); cl = QVBoxLayout(c); cl.setContentsMargins(0,0,0,0)
        cl.addWidget(self.banner); cl.addWidget(self.player, 1); left.addWidget(c)
        leftw = QWidget(); leftw.setLayout(left)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["시간", "이름", "반납/불출", "품목"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        rightw = QWidget(); rl = QVBoxLayout(rightw); rl.setContentsMargins(6,6,6,6); rl.addWidget(self.table)

        sp = QSplitter(); sp.addWidget(leftw); sp.addWidget(rightw); sp.setStretchFactor(0,3); sp.setStretchFactor(1,2)
        root = QWidget(); rl2 = QVBoxLayout(root); rl2.setContentsMargins(6,6,6,6); rl2.addWidget(sp)
        self.setCentralWidget(root)

        # 상태 머신
        self.hold_acc_s = 0.0   # 동시 인식 누적 시간
        self.reset_acc_s = 0.0  # 비충족 누적 시간
        self.armed = True       # 트리거 가능 상태
        self.last_t = 0.0
        self.window_names = deque(maxlen=90)  # 최근 구간 대표 라벨 다수결용
        self.window_items = deque(maxlen=90)

        self.ui_timer = QTimer(self); self.ui_timer.timeout.connect(self._on_tick)
        self.ui_timer.start(30)

    def _on_tick(self):
        t = self.player.time_s()
        dt = max(0.0, t - self.last_t)
        self.last_t = t
        ok, name, item = self.stream.get_state_at(t)

        if ok:
            self.hold_acc_s += dt
            self.reset_acc_s = 0.0
            if name: self.window_names.append(name)
            if item: self.window_items.append(item)
            if self.armed and self.hold_acc_s >= HOLD_MIN_S:
                rep_name = self._mode(self.window_names) or (name or "Unknown")
                rep_item = self._mode(self.window_items) or (item or "Unknown")
                log.info("Trigger @ %.2fs  name=%s  item=%s", t, rep_name, rep_item)
                self._trigger_modal(t, rep_name, rep_item)
        else:
            self.reset_acc_s += dt
            if self.reset_acc_s >= RESET_MIN_S:
                self.armed = True
                self.hold_acc_s = 0.0
                self.window_names.clear(); self.window_items.clear()

    def _mode(self, dq) -> Optional[str]:
        if not dq:
            return None
        return Counter(dq).most_common(1)[0][0]

    def _trigger_modal(self, t_s: float, person: str, item: str):
        self.player.pause()
        self.armed = False
        dlg = DecisionDialog(person, item, t_s, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.choice:
            self._append_decision(t_s, person, item, dlg.choice)
        self.player.play()

    def _append_decision(self, t_s: float, person: str, item: str, action: str):
        rec = {
            "timestamp_s": round(t_s, 3),
            "person_name": person,
            "item_class": item,
            "action": action,
        }
        try:
            with open(self.decisions_out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "쓰기 오류", f"decisions 저장 실패: {e}")
            return
        row = self.table.rowCount(); self.table.insertRow(row)
        ts_text = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.table.setItem(row, 0, QTableWidgetItem(ts_text))
        self.table.setItem(row, 1, QTableWidgetItem(person))
        self.table.setItem(row, 2, QTableWidgetItem(action))
        self.table.setItem(row, 3, QTableWidgetItem(item))
        self.table.scrollToBottom()

# ======================
# 엔트리 포인트
# ======================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    if not os.path.isfile(TEST_VIDEO_PATH):
        raise SystemExit(f"TEST_VIDEO_PATH 확인: {TEST_VIDEO_PATH}")
    if not os.path.isfile(TEST_RESULT_JSONL):
        raise SystemExit(f"TEST_RESULT_JSONL 확인: {TEST_RESULT_JSONL}")
    win = MainWindow(TEST_VIDEO_PATH, TEST_RESULT_JSONL, TEST_DECISIONS_JSONL)
    win.show()
    sys.exit(app.exec_())

