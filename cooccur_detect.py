#!/usr/bin/env python3
"""
cooccur_detect.py — Car + Bus Co-occurrence Detector  (standalone experiment)
==============================================================================
Fires an event clip whenever a CAR (COCO class 2) AND a BUS (COCO class 5)
are BOTH detected in the SAME frame simultaneously.

This is completely separate from main.py and does not touch the main pipeline.

Usage:
    python cooccur_detect.py
    python cooccur_detect.py --input input_video.mp4 --out-folder cooccur_clips
"""

import argparse
import collections
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

# Load root config.py directly (avoid config/ package shadow)
_spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).parent / "config.py"
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)
del _spec

# ── Constants ─────────────────────────────────────────────────────────
CAR_ID = 2
BUS_ID = 5
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck",
}
# Only detect cars and buses — nothing else
DETECT_CLASSES = [CAR_ID, BUS_ID]

# Pre/post roll and clip settings (reuse values from config)
PRE_ROLL_SECONDS  = config.PRE_ROLL_SECONDS
POST_ROLL_SECONDS = config.POST_ROLL_SECONDS
MAX_EVENT_SECONDS = config.MAX_EVENT_SECONDS
MIN_IDLE_SECONDS  = config.MIN_IDLE_SECONDS
MIN_BOX_AREA      = config.MIN_BOX_AREA


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resolve_input_path(path: str) -> str:
    """Resolve input video path with optional DATA_FOLDER fallback."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p)
    data_root = getattr(config, "DATA_FOLDER", None)
    if data_root:
        candidate = Path(data_root) / p
        if candidate.exists():
            return str(candidate)
    return str(p)


def _resolve_output_folder(folder: str) -> str:
    """Resolve output folder under config.OUTPUT_ROOT when set."""
    p = Path(folder)
    if p.is_absolute():
        return str(p)
    output_root = getattr(config, "OUTPUT_ROOT", None)
    if not output_root:
        return str(p)
    root = Path(output_root)
    try:
        p.relative_to(root)
        return str(p)
    except ValueError:
        return str(root / p)


def _open_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"),
        min(fps, config.OUTPUT_FPS), (w, h)
    )


def _encode(raw: str, final: str, ffmpeg_bin: str) -> None:
    h265 = ["-c:v", "libx265", "-preset", "fast", "-crf", "24", "-tag:v", "hvc1"]
    h264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    args = h265 if config.OUTPUT_CODEC == "h265" else h264
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw] + args + ["-movflags", "+faststart", final],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw] + h264 + ["-movflags", "+faststart", final],
            check=True, capture_output=True,
        )
    Path(raw).unlink(missing_ok=True)


def _annotate(frame: np.ndarray, boxes, video_time: float, event_num: int) -> np.ndarray:
    out = frame.copy()
    W = out.shape[1]
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color  = (0, 180, 255) if cls_id == CAR_ID else (255, 120, 0)
        name   = COCO_NAMES.get(cls_id, f"cls{cls_id}")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} {conf:.0%}", (x1, max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(out, (0, 0), (W, 26), (0, 30, 80), -1)
    cv2.putText(out, f"CAR+BUS EVENT #{event_num:03d}  {_fmt(video_time)}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    return out


def run(input_path: str, out_folder: str, device: str) -> None:
    input_path = _resolve_input_path(input_path)
    out_folder = _resolve_output_folder(out_folder)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pre_frames  = int(PRE_ROLL_SECONDS  * video_fps)
    post_frames = int(POST_ROLL_SECONDS * video_fps)
    max_frames  = int(MAX_EVENT_SECONDS * video_fps)
    gap_frames  = int(MIN_IDLE_SECONDS  * video_fps)
    interval    = config.DETECTION_INTERVAL   # frame-skip for speed

    clips_dir  = Path(out_folder)
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  Mode        : car + bus co-occurrence (same frame)")
    print(f"  Input       : {input_path}")
    print(f"  Clips saved : {clips_dir}/")
    print(f"  Video       : {width}×{height}  {video_fps:.1f}fps  "
          f"{_fmt(total_frames / video_fps)}")
    print(f"  Pre/Post    : {PRE_ROLL_SECONDS}s / {POST_ROLL_SECONDS}s"
          f"    Max clip: {MAX_EVENT_SECONDS}s    Gap: {MIN_IDLE_SECONDS}s")
    print(f"  Frame skip  : 1 in {interval}  (~{video_fps/interval:.0f}fps detection)")
    print(SEP)

    print("\n[YOLO] Loading model …")
    model = YOLO(config.YOLO_MODEL)
    model.to(device)
    print(f"[YOLO] Ready on {device}\n")

    # MOG2 for motion gate
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )

    pre_buf            = collections.deque(maxlen=pre_frames)
    event_active       = False
    event_num          = 0
    clips_written      = 0
    event_frame_count  = 0
    post_remaining     = 0
    frames_since_event = gap_frames
    motion_run         = 0
    frames_saved_total = 0
    frame_n            = 0
    n_yolo             = 0
    event_log          = []
    writer             = None
    raw_path = final_path = ""
    t_start  = time.time()

    def close_event(vt: float) -> None:
        nonlocal writer, clips_written
        if writer is None:
            return
        writer.release()
        writer = None
        try:
            _encode(raw_path, final_path, ffmpeg_bin)
            mb = Path(final_path).stat().st_size / 1_048_576
            print(f"    → clip saved: {Path(final_path).name}  ({mb:.1f} MB)")
            clips_written += 1
        except subprocess.CalledProcessError as e:
            print(f"    [WARN] encode failed: {e}")

    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        frame_n += 1

        if frame_n % 500 == 0:
            elapsed = time.time() - t_start
            fps_p   = frame_n / max(elapsed, 1e-9)
            eta     = (total_frames - frame_n) / max(fps_p, 1e-9)
            pct     = 100.0 * frame_n / max(total_frames, 1)
            print(f"  [{pct:5.1f}%]  {frame_n:,}/{total_frames:,}  "
                  f"{fps_p:.0f}fps  ETA {_fmt(eta)}  "
                  f"events={event_num}  clips_written={clips_written}")

        # Skip decoding between events for speed
        if not event_active and frame_n % interval != 0:
            frames_since_event += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            continue

        video_time = frame_n / video_fps

        # Motion gate
        fgmask   = subtractor.apply(frame)
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned  = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        has_motion = any(cv2.contourArea(c) >= config.MOTION_AREA_THRESHOLD
                         for c in contours)
        motion_run = (motion_run + 1) if has_motion else 0
        tier1 = motion_run >= config.MIN_MOTION_FRAMES
        pre_buf.append(frame.copy())

        if not tier1:
            if event_active:
                if post_remaining > 0:
                    writer.write(frame)
                    frames_saved_total += 1
                    post_remaining -= 1
                if post_remaining <= 0:
                    print(f"    → event ended at {_fmt(video_time)}")
                    close_event(video_time)
                    event_active = False
                    event_frame_count = 0
                    frames_since_event = 0
            else:
                frames_since_event += 1
            continue

        # YOLO: detect only cars and buses
        results = model.track(
            source=frame,
            conf=config.YOLO_CONFIDENCE,
            device=device,
            classes=DETECT_CLASSES,
            persist=True,
            verbose=False,
        )
        n_yolo += 1

        detected_classes = set()
        relevant_boxes   = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id      = int(box.cls[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue
                detected_classes.add(cls_id)
                relevant_boxes.append(box)

        # ── Trigger: BOTH car AND bus present in THIS frame ───────────
        in_gap  = (not event_active) and (frames_since_event < gap_frames)
        trigger = (
            not event_active
            and not in_gap
            and CAR_ID in detected_classes
            and BUS_ID in detected_classes
        )

        if trigger:
            event_num        += 1
            event_active      = True
            event_frame_count = 0
            post_remaining    = post_frames

            h = int(video_time // 3600)
            m = int((video_time % 3600) // 60)
            s = int(video_time % 60)
            fname      = f"event_{event_num:03d}__{h:02d}h{m:02d}m{s:02d}s__car_bus.mp4"
            final_path = str(clips_dir / fname)
            raw_path   = final_path + ".raw.mp4"
            writer     = _open_writer(raw_path, width, height, video_fps)

            print(f"\n  ★ EVENT #{event_num} at {_fmt(video_time)}"
                  f"  ←  car + bus simultaneously")
            event_log.append(
                f"EVENT #{event_num:03d}  {_fmt(video_time)}  car + bus simultaneously"
            )

            for buf_frame in pre_buf:
                writer.write(buf_frame)
                frames_saved_total += 1
            pre_buf.clear()

        if event_active:
            writer.write(_annotate(frame, relevant_boxes, video_time, event_num))
            frames_saved_total += 1
            event_frame_count  += 1
            if relevant_boxes:
                post_remaining = post_frames
            if event_frame_count >= max_frames:
                print(f"    → event capped at {_fmt(video_time)}")
                close_event(video_time)
                event_active = False
                event_frame_count = 0
                frames_since_event = 0
                post_remaining = 0
        else:
            frames_since_event += 1

    if event_active and writer is not None:
        print(f"    → event ended at {_fmt(video_time)}")
        close_event(video_time)

    cap.release()
    elapsed = time.time() - t_start

    # Size stats
    input_mb = Path(input_path).stat().st_size / 1_048_576
    clips_mb = sum(f.stat().st_size for f in clips_dir.glob("*.mp4")) / 1_048_576
    saved_mb = input_mb - clips_mb
    reduction_pct = 100.0 * (1.0 - clips_mb / max(input_mb * 1_048_576, 1) * 1_048_576 / max(input_mb * 1_048_576, 1))
    reduction_pct = 100.0 * (1.0 - clips_mb / max(input_mb, 1e-9))

    # Write event log
    log_path = clips_dir / "event_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"Car+Bus Co-occurrence Log — {input_path}\n")
        fh.write(f"Processed  : {_fmt(total_frames / video_fps)}"
                 f"  ({total_frames:,} frames @ {video_fps:.1f} fps)\n")
        fh.write(f"Events     : {event_num}\n\n")
        fh.write("── Size Reduction ─────────────────────────────────\n")
        fh.write(f"  Input video : {input_mb:>8.1f} MB\n")
        fh.write(f"  All clips   : {clips_mb:>8.1f} MB\n")
        fh.write(f"  Space saved : {saved_mb:>8.1f} MB  ({reduction_pct:.1f}% reduction)\n\n")
        fh.write("── Event Timeline ──────────────────────────────────\n")
        for entry in event_log:
            fh.write(entry + "\n")

    DSEP = "═" * 62
    print(f"\n{DSEP}")
    print(f"  Completed in     : {_fmt(elapsed)}")
    print(f"  Frames processed : {frame_n:,}")
    print(f"  YOLO calls       : {n_yolo:,}  ({100*n_yolo/max(frame_n,1):.1f}%)")
    print(f"  Car+bus events   : {event_num}")
    print(f"  Output folder    : {clips_dir}/")
    print(f"  Event log        : {log_path}")
    print(f"{'─'*62}")
    print(f"  ★ SIZE REDUCTION")
    print(f"    Input  : {input_mb:>7.1f} MB")
    print(f"    Clips  : {clips_mb:>7.1f} MB")
    print(f"    Saved  : {saved_mb:>7.1f} MB  →  {reduction_pct:.1f}% reduction")
    print(DSEP)
    if event_log:
        print("\n  ── Event Timeline ──")
        for e in event_log:
            print(f"    {e}")
    else:
        print("\n  No car+bus co-occurrence events found.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Car + Bus Co-occurrence Detector — fires when both are in the same frame"
    )
    ap.add_argument("--input",      default="input_video.mp4")
    ap.add_argument("--out-folder", default="cooccur_clips")
    ap.add_argument("--device",     default=config.YOLO_DEVICE)
    args = ap.parse_args()
    run(args.input, args.out_folder, args.device)


if __name__ == "__main__":
    main()
