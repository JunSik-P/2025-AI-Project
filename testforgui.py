#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_adapter.py

수집 후보 이벤트가 기록된 JSONL(…_events.jsonl)을 읽고,
영상 재생 중 해당 시점에 도달하면 일시정지를 트리거하도록 돕는 어댑터.

GUI/플레이어 프레임워크와 무관하게 사용할 수 있도록 설계.
- tick(current_ts): 이벤트 시점 도달 여부 확인(도달 시 이벤트 dict 반환 → 일시정지)
- commit(ev, decision): "반납"/"불출"/None(건너뛰기) 저장

입출력 경로(고정):
- INPUT : C:\Users\User\Desktop\csv\output\<video_stem>_events.jsonl
- OUTPUT: C:\Users\User\Desktop\csv\output\<video_stem>_decisions.jsonl
"""

from __future__ import annotations
import json, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ====== 고정 출력 루트 ======
OUTPUT_ROOT = Path(r"C:\Users\User\Desktop\csv\output")


# ====== JSONL 유틸 ======
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    if path.exists():
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    # 깨진 라인은 무시
                    pass
    return rows

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ====== 어댑터 본체 ======
class DecisionAdapter:
    """
    사용 시나리오(의사코드):
        adapter = DecisionAdapter(video_path)
        while playing:
            ts = player.current_time_seconds()  # 현재 재생 시각(초)
            ev = adapter.tick(ts)
            if ev:
                player.pause()
                ui.show_modal(
                    title="수집 사건 분류",
                    text=f"{ev['timestamp_s']:.2f}s / {ev.get('person_name')} / {ev.get('item_class')}",
                    buttons={
                        "반납": lambda: (adapter.commit(ev, "반납"), player.resume()),
                        "불출": lambda: (adapter.commit(ev, "불출"), player.resume()),
                        "건너뛰기": lambda: (adapter.commit(ev, None), player.resume())
                    }
                )
    """

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        base = self.video_path.stem

        self.events_path    = OUTPUT_ROOT / f"{base}_events.jsonl"
        self.decisions_path = OUTPUT_ROOT / f"{base}_decisions.jsonl"

        # 이벤트 로드(+필터) & 시간순 정렬
        self.events: List[Dict[str, Any]] = self._load_and_filter_events(self.events_path)
        self.events.sort(key=lambda e: float(e.get("timestamp_s", 0.0)))

        # 이미 내려진 결정 캐시(중복 방지)
        self.decisions_cache: Dict[Tuple[float,int,int], str] = self._load_decisions_cache(self.decisions_path)

        # 진행 상태
        self._idx: int = 0          # 다음 검사할 이벤트 인덱스
        self._pending: Optional[Dict[str, Any]] = None  # 현재 질문 중인 이벤트

    # ---------- 내부 로더 ----------
    def _load_and_filter_events(self, path: Path) -> List[Dict[str, Any]]:
        """
        필터 규칙:
          - person_name == "Unknown" 제외
          - status == "needs_face_front" 제외
          - timestamp_s 필수
        """
        evs = _load_jsonl(path)
        out = []
        for e in evs:
            pname  = str(e.get("person_name", "Unknown")).lower()
            status = e.get("status", "pending")
            if pname == "unknown":
                continue
            if status == "needs_face_front":
                continue
            if "timestamp_s" not in e:
                continue
            out.append(e)
        return out

    def _load_decisions_cache(self, path: Path) -> Dict[Tuple[float,int,int], str]:
        cache: Dict[Tuple[float,int,int], str] = {}
        for r in _load_jsonl(path):
            key = (float(r.get("timestamp_s", 0.0)),
                   int(r.get("person_track_id", -1)),
                   int(r.get("item_track_id", -1)))
            dec = r.get("decision")
            if dec in ("반납","불출"):
                cache[key] = dec
        return cache

    def _key(self, e: Dict[str, Any]) -> Tuple[float,int,int]:
        return (float(e.get("timestamp_s", 0.0)),
                int(e.get("person_track_id", -1)),
                int(e.get("item_track_id", -1)))

    def _is_done(self, e: Dict[str, Any]) -> bool:
        return self._key(e) in self.decisions_cache

    # ---------- 외부 인터페이스 ----------
    def tick(self, current_ts: float, early_threshold: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        재생 루프마다 호출.
        - current_ts: 현재 재생 시각(초)
        - 반환값: 이벤트 dict (일시정지 트리거). 없으면 None.

        early_threshold: 이벤트 시각보다 최대 얼마 앞서 질문을 띄울지(초)
        """
        # 이미 질문 중인 이벤트가 있으면 그대로 반환(중복 질문 방지)
        if self._pending is not None:
            return self._pending

        n = len(self.events)

        # 인덱스 전진(이미 처리된 이벤트 스킵)
        while self._idx < n and self._is_done(self.events[self._idx]):
            self._idx += 1

        if self._idx >= n:
            return None

        ev = self.events[self._idx]
        ev_ts = float(ev.get("timestamp_s", 0.0))

        # 이벤트 시점 도달(약간 앞서 허용)
        if current_ts >= ev_ts - early_threshold:
            self._pending = ev
            return ev
        return None

    def commit(self, ev: Dict[str, Any], decision: Optional[str]) -> None:
        """
        GUI에서 사용자가 선택했을 때 호출:
          - decision: "반납" | "불출" | None(건너뛰기)
        """
        ts, p, i = self._key(ev)
        rec = {
            "timestamp_s": ts,
            "person_track_id": p,
            "item_track_id": i,
            "person_name": ev.get("person_name", None),
            "item_class": ev.get("item_class", None),
            "decision": decision,              # "반납" | "불출" | None
            "decided_at": time.time()
        }
        _append_jsonl(self.decisions_path, rec)

        # 캐시 갱신 & 상태 갱신
        if decision in ("반납","불출"):
            self.decisions_cache[(ts, p, i)] = decision
        self._pending = None
        self._idx += 1

    # ---------- (선택) 시킹 대응 ----------
    def resync_after_seek(self, current_ts: float) -> None:
        """
        타임라인을 크게 이동(앞/뒤)했을 때 호출하면,
        current_ts 이전의 처리되지 않은 이벤트들을 건너뛴 위치로 인덱스를 조정.
        """
        n = len(self.events)
        # 처리된 이벤트는 건너뛰고,
        # 처리되지 않았더라도 현재 시간보다 충분히 이전이면 스킵하도록 전진
        while self._idx < n:
            ev = self.events[self._idx]
            ev_ts = float(ev.get("timestamp_s", 0.0))
            if self._is_done(ev):
                self._idx += 1
                continue
            if ev_ts < current_ts - 0.25:  # 0.25초 버퍼
                self._idx += 1
                continue
            break
        # 진행 중이던 질문 상태 초기화
        self._pending = None


# ====== (옵션) 간단 데모 러너 ======
if __name__ == "__main__":
    """
    이 블록은 팀원이 독립 실행으로 동작 확인할 수 있는 미니 데모입니다.
    - 실제 GUI 플레이어가 있으면 이 부분은 사용하지 않아도 됩니다.
    - 영상 재생/일시정지는 구현하지 않고, 콘솔에서 tick → commit 흐름만 보여줍니다.
    사용법:
        python decision_adapter.py "C:\path\to\video.mp4"
    """
    import sys, time as _time
    if len(sys.argv) < 2:
        print("usage: python decision_adapter.py <video_path>")
        sys.exit(0)

    vpath = Path(sys.argv[1])
    adapter = DecisionAdapter(vpath)

    # 가짜 재생 루프(초 단위로 0.1씩 증가)
    cur = 0.0
    end = 3600.0  # 1시간 가드
    print(f"[demo] events loaded: {len(adapter.events)} from {adapter.events_path}")
    while cur < end:
        ev = adapter.tick(cur)
        if ev:
            print(f"\n[PAUSE] due at t={ev['timestamp_s']:.2f}s | "
                  f"{ev.get('person_name')} / {ev.get('item_class')} | "
                  f"tracks p={ev.get('person_track_id')} i={ev.get('item_track_id')}")
            # 콘솔 입력으로 결정
            s = input("결정 입력 (r=반납 / i=불출 / s=건너뛰기): ").strip().lower()
            dec = {"r":"반납", "i":"불출", "s":None}.get(s, None)
            adapter.commit(ev, dec)
            print(f"[RESUME] decision saved: {dec}")
        cur += 0.1
        _time.sleep(0.02)

    print(f"[demo] decisions saved to: {adapter.decisions_path}")
