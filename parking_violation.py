"""
离线视频违停检测流水线：Faster R-CNN 检测 → IoU 多目标关联 → ROI(多边形)中心点判定 → 停留计时与事件导出。

用法（在项目根目录）:
  python illegal_parking_pipeline.py --video path/to/video.mp4 --roi configs/roi.json --out_dir outputs/run1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from detection_single import build_model, to_tensor_rgb01


# ---------------------------------------------------------------------------
# 几何：多边形内点、IoU
# ---------------------------------------------------------------------------


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """射线法，poly 为 [[x,y], ...] 闭合或不必重复首点。"""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def bbox_center_xyxy(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box.tolist()
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_greedy_iou(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    iou_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """按 IoU 降序贪心匹配，返回 (pairs, unmatched_track_idx, unmatched_det_idx)。"""
    nt, nd = track_boxes.shape[0], det_boxes.shape[0]
    if nt == 0 or nd == 0:
        return [], list(range(nt)), list(range(nd))

    pairs: List[Tuple[int, int]] = []
    cand: List[Tuple[float, int, int]] = []
    for i in range(nt):
        for j in range(nd):
            iou = iou_xyxy(track_boxes[i], det_boxes[j])
            if iou >= iou_thresh:
                cand.append((iou, i, j))
    cand.sort(key=lambda t: -t[0])

    used_t = set()
    used_d = set()
    for _, i, j in cand:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        pairs.append((i, j))

    unmatched_t = [i for i in range(nt) if i not in used_t]
    unmatched_d = [j for j in range(nd) if j not in used_d]
    return pairs, unmatched_t, unmatched_d


# ---------------------------------------------------------------------------
# 跟踪：IoU 关联 + 丢检容忍帧数
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    box_xyxy: np.ndarray
    time_since_update: int = 0
    hits: int = 1


class IoUTracker:
    def __init__(self, iou_thresh: float, max_age: int):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._next_id = 1
        self.tracks: List[Track] = []

    def update(self, det_boxes: np.ndarray) -> Tuple[Dict[int, np.ndarray], Set[int], Set[int]]:
        """
        det_boxes: [N, 4]
        返回 (track_id -> 当前框, 本帧由检测更新的 id 集合, 本帧被删除的 id 集合)。
        未匹配但仍在 max_age 内的轨迹保留上一帧框，用于短时丢检容忍。
        """
        matched_ids: Set[int] = set()
        if det_boxes.shape[0] == 0:
            before = {t.track_id for t in self.tracks}
            for t in self.tracks:
                t.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            removed = before - {t.track_id for t in self.tracks}
            return {t.track_id: t.box_xyxy.copy() for t in self.tracks}, matched_ids, removed

        if len(self.tracks) == 0:
            for j in range(det_boxes.shape[0]):
                tid = self._next_id
                self._next_id += 1
                self.tracks.append(Track(tid, det_boxes[j].copy()))
                matched_ids.add(tid)
            return {t.track_id: t.box_xyxy.copy() for t in self.tracks}, matched_ids, set()

        before = {t.track_id for t in self.tracks}
        track_boxes = np.stack([t.box_xyxy for t in self.tracks], axis=0)
        pairs, unmatched_t, unmatched_d = match_greedy_iou(track_boxes, det_boxes, self.iou_thresh)

        for ti, dj in pairs:
            tr = self.tracks[ti]
            tr.box_xyxy = det_boxes[dj].copy()
            tr.time_since_update = 0
            tr.hits += 1
            matched_ids.add(tr.track_id)

        for ti in unmatched_t:
            self.tracks[ti].time_since_update += 1

        for dj in unmatched_d:
            tid = self._next_id
            self._next_id += 1
            self.tracks.append(Track(tid, det_boxes[dj].copy()))
            matched_ids.add(tid)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        removed = before - {t.track_id for t in self.tracks}
        out = {t.track_id: t.box_xyxy.copy() for t in self.tracks}
        return out, matched_ids, removed


# ---------------------------------------------------------------------------
# ROI 配置加载
# ---------------------------------------------------------------------------


def load_rois(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rois = data.get("rois", data)
    if not isinstance(rois, list):
        raise ValueError("roi.json 应包含 'rois' 数组")
    out = []
    for r in rois:
        rid = r.get("id", str(len(out)))
        poly = r.get("polygon")
        if not poly or len(poly) < 3:
            continue
        out.append({"id": str(rid), "name": r.get("name", ""), "polygon": poly})
    if not out:
        raise ValueError("roi.json 中没有有效 polygon（至少 3 个点）")
    return out


def roi_for_point(x: float, y: float, rois: List[Dict[str, Any]]) -> Optional[str]:
    for r in rois:
        if point_in_polygon(x, y, r["polygon"]):
            return str(r["id"])
    return None


# ---------------------------------------------------------------------------
# 违停计时与事件
# ---------------------------------------------------------------------------


@dataclass
class DwellState:
    roi_id: Optional[str] = None
    segment_start_frame: int = 0
    segment_start_sec: float = 0.0
    accumulated_sec: float = 0.0
    violation_emitted: bool = False
    lost_frames: int = 0


@dataclass
class ViolationEvent:
    event_id: int
    track_id: int
    roi_id: str
    start_frame: int
    trigger_frame: int
    end_frame: Optional[int]
    start_sec: float
    trigger_sec: float
    end_sec: Optional[float]
    duration_sec: float
    snapshot_path: Optional[str] = None


@dataclass
class PipelineState:
    dwell: Dict[int, DwellState] = field(default_factory=dict)
    events: List[ViolationEvent] = field(default_factory=list)
    _next_event_id: int = 1

    def ensure_dwell(self, tid: int) -> DwellState:
        if tid not in self.dwell:
            self.dwell[tid] = DwellState()
        return self.dwell[tid]


def finalize_roi_dwell(
    tid: int,
    frame_idx: int,
    fps: float,
    ps: PipelineState,
    violation_open: Dict[int, ViolationEvent],
) -> None:
    """结束当前 ROI 停留段（离开 ROI、切换 ROI、丢检超时或视频结束）。"""
    st = ps.dwell.get(tid)
    if st is None or st.roi_id is None:
        return
    if tid in violation_open:
        ev = violation_open.pop(tid)
        ev.end_frame = frame_idx
        ev.end_sec = frame_idx / fps
        ev.duration_sec = max(0.0, (ev.end_sec or 0) - ev.start_sec)
    st.roi_id = None
    st.segment_start_frame = 0
    st.segment_start_sec = 0.0
    st.accumulated_sec = 0.0
    st.violation_emitted = False
    st.lost_frames = 0


def try_emit_violation(
    frame_idx: int,
    fps: float,
    tid: int,
    roi_id: str,
    st: DwellState,
    threshold_sec: float,
    ps: PipelineState,
    snapshots_dir: Optional[Path],
    violation_open: Dict[int, ViolationEvent],
) -> None:
    if st.accumulated_sec < threshold_sec or st.violation_emitted:
        return
    st.violation_emitted = True
    trig_sec = frame_idx / fps
    ev = ViolationEvent(
        event_id=ps._next_event_id,
        track_id=tid,
        roi_id=roi_id,
        start_frame=st.segment_start_frame,
        trigger_frame=frame_idx,
        end_frame=None,
        start_sec=st.segment_start_sec,
        trigger_sec=trig_sec,
        end_sec=None,
        duration_sec=st.accumulated_sec,
        snapshot_path=None,
    )
    ps._next_event_id += 1
    ps.events.append(ev)
    violation_open[tid] = ev
    if snapshots_dir is not None:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap = snapshots_dir / f"violation_{ev.event_id:04d}_f{frame_idx:06d}_tid{tid}.jpg"
        ev.snapshot_path = str(snap)


def update_dwell_matched_track(
    frame_idx: int,
    fps: float,
    dt: float,
    threshold_sec: float,
    rois: List[Dict[str, Any]],
    tid: int,
    box: np.ndarray,
    ps: PipelineState,
    snapshots_dir: Optional[Path],
    violation_open: Dict[int, ViolationEvent],
) -> None:
    """本帧有检测关联：按 bbox 中心点更新 ROI 与停留时间。"""
    cx, cy = bbox_center_xyxy(box)
    rid = roi_for_point(cx, cy, rois)
    st = ps.ensure_dwell(tid)
    st.lost_frames = 0

    if rid is None:
        finalize_roi_dwell(tid, frame_idx, fps, ps, violation_open)
        return

    if st.roi_id is None or st.roi_id != rid:
        finalize_roi_dwell(tid, frame_idx, fps, ps, violation_open)
        st.roi_id = rid
        st.segment_start_frame = frame_idx
        st.segment_start_sec = frame_idx / fps
        st.accumulated_sec = 0.0
        st.violation_emitted = False

    st.accumulated_sec += dt
    try_emit_violation(frame_idx, fps, tid, rid, st, threshold_sec, ps, snapshots_dir, violation_open)


def update_dwell_coasting_track(
    frame_idx: int,
    fps: float,
    dt: float,
    gap_frames: int,
    threshold_sec: float,
    tid: int,
    time_since_update: int,
    ps: PipelineState,
    snapshots_dir: Optional[Path],
    violation_open: Dict[int, ViolationEvent],
) -> None:
    """本帧无检测、沿用上一框：在 ROI 段内按丢检容忍继续计时，超时则结束段。"""
    st = ps.dwell.get(tid)
    if st is None or st.roi_id is None:
        return
    st.lost_frames = time_since_update
    rid = st.roi_id
    if st.lost_frames <= gap_frames:
        st.accumulated_sec += dt
        try_emit_violation(frame_idx, fps, tid, rid, st, threshold_sec, ps, snapshots_dir, violation_open)
    else:
        finalize_roi_dwell(tid, frame_idx, fps, ps, violation_open)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def draw_overlay(
    frame_rgb: np.ndarray,
    rois: List[Dict[str, Any]],
    track_id_to_box: Dict[int, np.ndarray],
    violation_open: Dict[int, ViolationEvent],
    font: Optional[Any],
) -> Image.Image:
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    for r in rois:
        poly = [(float(p[0]), float(p[1])) for p in r["polygon"]]
        if len(poly) >= 2:
            draw.line(poly + [poly[0]], fill=(0, 255, 255), width=2)
            if font:
                draw.text((poly[0][0], max(0, poly[0][1] - 14)), str(r["id"]), fill=(0, 255, 255), font=font)
    for tid, box in track_id_to_box.items():
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        color = (255, 0, 0) if tid in violation_open else (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"id{tid}"
        if tid in violation_open:
            label += " VIOL"
        if font:
            draw.text((x1, max(0, y1 - 12)), label, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        else:
            draw.text((x1, max(0, y1 - 12)), label, fill=color)
    return img


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    import cv2
    import os

    print("Starting run_pipeline...")
    video_path = Path(args.video)
    roi_path = Path(args.roi)
    out_dir = Path(args.out_dir)
    
    print(f"Video path: {video_path}")
    print(f"ROI path: {roi_path}")
    print(f"Output directory: {out_dir}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {out_dir.exists()}")
    
    snap_dir = out_dir / "snapshots" if args.save_snapshots else None
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)
        print(f"Snapshots directory created: {snap_dir.exists()}")

    print("Loading ROIs...")
    rois = load_rois(roi_path)
    print(f"Loaded {len(rois)} ROIs")
    
    print("Opening video capture...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    print("Video capture opened successfully")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = float(args.fps_fallback)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gap_frames = max(0, int(round(args.gap_tolerance_seconds * fps)))
    dt = 1.0 / fps

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 2))
    model = build_model(num_classes=num_classes, min_size=args.min_size, max_size=args.max_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tracker = IoUTracker(iou_thresh=args.track_iou_thresh, max_age=gap_frames + 1)
    ps = PipelineState()
    violation_open: Dict[int, ViolationEvent] = {}

    overlay_path = out_dir / "overlay.mp4"
    # 尝试不同的编码格式，提高兼容性
    fourccs = [
        cv2.VideoWriter_fourcc(*"mp4v"),  # 标准 MP4 编码
        cv2.VideoWriter_fourcc(*"avc1"),  # H.264 编码
        cv2.VideoWriter_fourcc(*"xvid"),  # XVID 编码
    ]
    
    writer = None
    for fourcc in fourccs:
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"成功创建 VideoWriter，使用编码: {chr(fourcc & 0xFF)}{chr((fourcc >> 8) & 0xFF)}{chr((fourcc >> 16) & 0xFF)}{chr((fourcc >> 24) & 0xFF)}")
            break
    
    if not writer or not writer.isOpened():
        raise SystemExit("无法创建 VideoWriter，请检查 OpenCV 与编码支持")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    frame_idx = 0

    with torch.no_grad():
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            x = to_tensor_rgb01(pil).to(device)
            out = model([x])[0]
            boxes_t = out["boxes"]
            scores_t = out["scores"]
            keep = scores_t >= args.score_thresh
            boxes_t = boxes_t[keep].cpu().numpy()
            scores_t = scores_t[keep].cpu().numpy()

            det_boxes = boxes_t.astype(np.float32) if boxes_t.size else np.zeros((0, 4), dtype=np.float32)
            track_id_to_box, matched_ids, removed_ids = tracker.update(det_boxes)

            for tid in removed_ids:
                finalize_roi_dwell(tid, frame_idx, fps, ps, violation_open)

            tsu_by_id = {t.track_id: t.time_since_update for t in tracker.tracks}

            for tid, box in track_id_to_box.items():
                if tid in matched_ids:
                    update_dwell_matched_track(
                        frame_idx,
                        fps,
                        dt,
                        args.threshold_seconds,
                        rois,
                        tid,
                        box,
                        ps,
                        snap_dir,
                        violation_open,
                    )
                else:
                    update_dwell_coasting_track(
                        frame_idx,
                        fps,
                        dt,
                        gap_frames,
                        args.threshold_seconds,
                        tid,
                        tsu_by_id[tid],
                        ps,
                        snap_dir,
                        violation_open,
                    )

            if snap_dir is not None:
                for tid, ev in list(violation_open.items()):
                    if ev.snapshot_path and tid in track_id_to_box and not Path(ev.snapshot_path).exists():
                        vis = draw_overlay(frame_rgb, rois, {tid: track_id_to_box[tid]}, violation_open, font)
                        vis.convert("RGB").save(ev.snapshot_path, quality=95)

            overlay = draw_overlay(frame_rgb, rois, track_id_to_box, violation_open, font)
            writer.write(cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))

            frame_idx += 1

    cap.release()
    writer.release()

    # 尝试使用 moviepy 合并原始音频
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        # 读取原始视频的音频
        original_clip = VideoFileClip(str(video_path))
        audio = original_clip.audio
        
        if audio:
            # 读取处理后的视频
            processed_clip = VideoFileClip(str(overlay_path))
            
            # 合并音频
            final_clip = processed_clip.set_audio(audio)
            
            # 输出最终视频（临时文件）
            temp_path = out_dir / "temp_overlay.mp4"
            final_clip.write_videofile(str(temp_path), codec='libx264', audio_codec='aac')
            
            # 替换原始文件
            import os
            os.replace(str(temp_path), str(overlay_path))
            
            print("音频已成功合并到输出视频")
        else:
            print("原始视频没有音频轨道")
    except Exception as e:
        print(f"音频处理失败: {e}")
        print("将使用无音频的视频文件")

    for tid in list(ps.dwell.keys()):
        st = ps.dwell.get(tid)
        if st and st.roi_id is not None:
            finalize_roi_dwell(tid, frame_idx, fps, ps, violation_open)

    events_path = out_dir / "events.json"
    events_serial = [
        {
            "event_id": e.event_id,
            "track_id": e.track_id,
            "roi_id": e.roi_id,
            "start_frame": e.start_frame,
            "trigger_frame": e.trigger_frame,
            "end_frame": e.end_frame,
            "start_sec": round(e.start_sec, 4),
            "trigger_sec": round(e.trigger_sec, 4),
            "end_sec": None if e.end_sec is None else round(e.end_sec, 4),
            "duration_sec": round(e.duration_sec, 4),
            "snapshot_path": e.snapshot_path,
        }
        for e in ps.events
    ]
    events_path.write_text(json.dumps(events_serial, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "events.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            [
                "event_id",
                "track_id",
                "roi_id",
                "start_frame",
                "trigger_frame",
                "end_frame",
                "start_sec",
                "trigger_sec",
                "end_sec",
                "duration_sec",
                "snapshot_path",
            ]
        )
        for e in ps.events:
            wcsv.writerow(
                [
                    e.event_id,
                    e.track_id,
                    e.roi_id,
                    e.start_frame,
                    e.trigger_frame,
                    e.end_frame,
                    f"{e.start_sec:.4f}",
                    f"{e.trigger_sec:.4f}",
                    "" if e.end_sec is None else f"{e.end_sec:.4f}",
                    f"{e.duration_sec:.4f}",
                    e.snapshot_path or "",
                ]
            )

    meta = {
        "video": str(video_path),
        "roi": str(roi_path),
        "fps": fps,
        "frames": frame_idx,
        "threshold_seconds": args.threshold_seconds,
        "gap_tolerance_seconds": args.gap_tolerance_seconds,
        "score_thresh": args.score_thresh,
        "num_events": len(ps.events),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成: 帧数={frame_idx} | 违停事件数={len(ps.events)} | 输出目录={out_dir}")


def parse_args() -> argparse.Namespace:
    # 配置参数 - 用户可以直接修改这些默认值
    config = {
        "video": None,  # 输入视频路径（通过命令行参数传入）
        "roi": None,     # ROI 多边形 JSON（通过命令行参数传入）
        "ckpt": Path("D:/毕业设计/bysj/bysj/runs/fasterrcnn_vehicle_resnet_best_aliyun.pt"),  # 检测器权重
        "out_dir": Path("D:/毕业设计/bysj/bysj/outputs/result"),  # 输出目录
        "threshold_seconds": 6.0,  # ROI 内停留超过该秒数判违停
        "gap_tolerance_seconds": 1.5,  # 短时丢检容忍（秒）
        "fps_fallback": 25.0,  # 视频无 FPS 元数据时的假定帧率
        "score_thresh": 0.5,  # 检测分数阈值
        "track_iou_thresh": 0.3,  # 跟踪 IoU 匹配阈值
        "min_size": 800,
        "max_size": 1333,
        "save_snapshots": True,  # 违停触发帧保存截图到 snapshots/
        "cpu": False
    }
    
    ap = argparse.ArgumentParser(description="违停检测：检测→跟踪→ROI→计时→导出")
    ap.add_argument("--video", type=Path, required=True, help="输入视频路径")
    ap.add_argument("--roi", type=Path, required=True, help="ROI 多边形 JSON")
    ap.add_argument("--ckpt", type=Path, default=config["ckpt"], help="检测器权重")
    ap.add_argument("--out_dir", type=Path, default=config["out_dir"], help="输出目录")
    ap.add_argument("--threshold_seconds", type=float, default=config["threshold_seconds"], help="ROI 内停留超过该秒数判违停")
    ap.add_argument("--gap_tolerance_seconds", type=float, default=config["gap_tolerance_seconds"], help="短时丢检容忍（秒）")
    ap.add_argument("--fps_fallback", type=float, default=config["fps_fallback"], help="视频无 FPS 元数据时的假定帧率")
    ap.add_argument("--score_thresh", type=float, default=config["score_thresh"], help="检测分数阈值")
    ap.add_argument("--track_iou_thresh", type=float, default=config["track_iou_thresh"], help="跟踪 IoU 匹配阈值")
    ap.add_argument("--min_size", type=int, default=config["min_size"])
    ap.add_argument("--max_size", type=int, default=config["max_size"])
    # store_true 参数的处理：如果 config 中为 True，则直接设置为默认启用
    if config["save_snapshots"]:
        ap.add_argument("--no_save_snapshots", action="store_false", dest="save_snapshots", help="不保存违停触发帧截图")
    else:
        ap.add_argument("--save_snapshots", action="store_true", help="违停触发帧保存截图到 snapshots/")
    
    if config["cpu"]:
        ap.add_argument("--no_cpu", action="store_false", dest="cpu", help="使用 GPU")
    else:
        ap.add_argument("--cpu", action="store_true", help="使用 CPU")
    return ap.parse_args()


def main() -> None:
    print("Starting main function...")
    try:
        import cv2  # noqa: F401
        print("cv2 imported successfully")
    except ImportError:
        raise SystemExit("请先安装 opencv-python-headless: pip install opencv-python-headless")
    
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        print("moviepy imported successfully")
    except ImportError:
        print("moviepy not installed, audio will not be processed")
    
    print("Parsing arguments...")
    args = parse_args()
    print(f"Arguments parsed: video={args.video}, roi={args.roi}, out_dir={args.out_dir}")
    print(f"Video file exists: {args.video.exists()}")
    print(f"ROI file exists: {args.roi.exists()}")
    print(f"Checkpoint file exists: {args.ckpt.exists()}")
    print("Running pipeline...")
    run_pipeline(args)
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
