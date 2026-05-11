"""
test_cooccurrence.py  —  EXPERIMENTAL / THROWAWAY
==================================================
Event = car AND bus appear IN THE SAME FRAME together.

Clips are saved to  cooccur_clips/
Nothing in main.py or config.py is touched.

Usage:
    python test_cooccurrence.py
    python test_cooccurrence.py --input input_video.mp4 --out-folder cooccur_clips
"""

import argparse
import collections
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

import config   # re-use model/device/thresholds — no writes to config

# ── Target classes for this experiment ───────────────────────────────
CAR_ID = 2
BUS_ID = 5
CO_OCCUR_CLASSES = {CAR_ID, BUS_ID}   # both must be present to trigger

# Clip folder separate from production output
DEFAULT_OUT_FOLDER = "cooccur_clips"

# Padding (seconds)
PRE_ROLL  = 4
POST_ROLL = 5
MAX_CLIP  = 40   # hard cap seconds
MIN_GAP   = 15   # minimum seconds between events

# ── Colour palette for annotations ───────────────────────────────────
COLORS = {
    CAR_ID: (0,  180, 255),   # sky-blue  — car
    BUS_ID: (180,  0, 255),   # purple    — bus
}
CLASS_NAMES = {CAR_ID: "car", BUS_ID: "bus"}


# ─────────────────────────────────────────────────────────────────────

def fmt(s: float) -> str:
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"


def clip_name(event_num: int, video_time: float) -> str:
    h = int(video_time // 3600)
    m = int((video_time % 3600) // 60)
    s = int(video_time % 60)
    return f"cooccur_{event_num:03d}__{h:02d}h{m:02d}m{s:02d}s__car_and_bus.mp4"


def open_writer(path: str, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, config.OUTPUT_FPS, (w, h))


def encode_h264(raw: str, out: str, ffmpeg_bin: str) -> None:
    subprocess.run([
        ffmpeg_bin, "-y", "-i", raw,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-movflags", "+faststart", out
    ], check=True, capture_output=True)
    Path(raw).unlink(missing_ok=True)


def annotate(frame: np.ndarray, boxes, video_time: float) -> np.ndarray:
    vis = frame.copy()
    w   = vis.shape[1]
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls_id,'?')}  {conf:.0%}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(vis, (0, 0), (w, 28), (0, 60, 0), -1)
    cv2.putText(vis, f"★ CAR+BUS  {fmt(video_time)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return vis


# ─────────────────────────────────────────────────────────────────────

def run(input_path: str, out_folder: str) -> None:
    # Resolve input/output locations using the shared config.
    p_in = Path(input_path)
    if not p_in.is_absolute() and not p_in.exists():
        data_root = getattr(config, "DATA_FOLDER", None)
        if data_root:
            candidate = Path(data_root) / p_in
            if candidate.exists():
                input_path = str(candidate)

    p_out = Path(out_folder)
    if not p_out.is_absolute():
        out_root = getattr(config, "OUTPUT_ROOT", None)
        if out_root:
            root = Path(out_root)
            try:
                p_out.relative_to(root)
            except ValueError:
                p_out = root / p_out
        out_folder = str(p_out)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pre_frames  = int(PRE_ROLL  * video_fps)
    post_frames = int(POST_ROLL * video_fps)
    max_frames  = int(MAX_CLIP  * video_fps)
    gap_frames  = int(MIN_GAP   * video_fps)

    clips_dir = Path(out_folder)
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"\n{'─'*60}")
    print(f"  EXPERIMENT: car+bus co-occurrence clips")
    print(f"  Input  : {input_path}")
    print(f"  Output : {clips_dir}/")
    print(f"  Video  : {width}×{height}  {video_fps:.1f}fps  {fmt(total_frames/video_fps)}")
    print(f"{'─'*60}\n")

    # ── Background subtractor (Tier-1 gate) ───────────────────────────
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )

    # ── YOLO model ────────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8 model …")
    model = YOLO(config.YOLO_MODEL)
    model.to(config.YOLO_DEVICE)
    print(f"[INFO] Model ready on {config.YOLO_DEVICE}\n")

    # ── State ─────────────────────────────────────────────────────────
    pre_buf: collections.deque = collections.deque(maxlen=pre_frames)
    event_active       = False
    event_num          = 0
    event_frame_count  = 0
    post_remaining     = 0
    frames_since_event = gap_frames      # allow event immediately at start
    motion_run         = 0

    writer: cv2.VideoWriter | None = None
    raw_path   = ""
    final_path = ""

    n_read = n_yolo = 0
    event_log: list[str] = []

    t_start = time.time()

    def close_event(video_time: float) -> None:
        nonlocal writer, raw_path, final_path
        if writer is None:
            return
        writer.release()
        writer = None
        try:
            encode_h264(raw_path, final_path, ffmpeg_bin)
            size_mb = Path(final_path).stat().st_size / 1024 ** 2
            print(f"    → clip saved: {Path(final_path).name}  ({size_mb:.1f} MB)")
        except subprocess.CalledProcessError as e:
            print(f"    [WARN] ffmpeg failed: {e}")

    # ── Frame loop ────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        n_read    += 1
        video_time = n_read / video_fps

        # Progress every 500 frames
        if n_read % 500 == 0:
            elapsed = time.time() - t_start
            fps_p   = n_read / max(elapsed, 1e-6)
            eta     = (total_frames - n_read) / max(fps_p, 1e-6)
            pct     = 100 * n_read / max(total_frames, 1)
            print(f"  [{pct:5.1f}%]  {n_read:,}/{total_frames:,}  "
                  f"{fps_p:.0f}fps  ETA {fmt(eta)}  events={event_num}")

        # Tier-1: motion gate
        fgmask = subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) >= config.MOTION_AREA_THRESHOLD
                     for c in contours)

        if motion:
            motion_run += 1
        else:
            motion_run  = 0

        tier1 = motion_run >= config.MIN_MOTION_FRAMES
        pre_buf.append(frame.copy())

        if not tier1:
            if event_active:
                if post_remaining > 0:
                    writer.write(frame)
                    post_remaining -= 1
                if post_remaining <= 0:
                    print(f"    → event ended at {fmt(video_time)}")
                    close_event(video_time)
                    event_active      = False
                    event_frame_count = 0
                    frames_since_event = 0
            else:
                frames_since_event += 1
            continue

        # Tier-2: YOLO — only detect car and bus
        results = model.predict(
            source=frame,
            conf=config.YOLO_CONFIDENCE,
            device=config.YOLO_DEVICE,
            classes=list(CO_OCCUR_CLASSES),
            verbose=False,
        )
        n_yolo += 1

        detected_classes = set()
        relevant_boxes   = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                detected_classes.add(cls_id)
                relevant_boxes.append(box)

        # ── Core rule: BOTH car AND bus must be visible ───────────────
        co_occur = CO_OCCUR_CLASSES.issubset(detected_classes)

        in_gap = (not event_active) and (frames_since_event < gap_frames)

        if co_occur and not event_active and not in_gap:
            event_num        += 1
            event_active      = True
            event_frame_count = 0
            post_remaining    = post_frames

            print(f"\n  ★ CAR+BUS EVENT #{event_num} at {fmt(video_time)}")
            event_log.append(f"EVENT #{event_num:03d}  {fmt(video_time)}  car+bus co-occurrence")

            fname      = clip_name(event_num, video_time)
            final_path = str(clips_dir / fname)
            raw_path   = final_path + ".raw.mp4"
            writer     = open_writer(raw_path, width, height)

            # pre-roll
            for buf_frame in pre_buf:
                writer.write(buf_frame)
            pre_buf.clear()

        if event_active:
            # write annotated frame so clips are easy to review
            writer.write(annotate(frame, relevant_boxes, video_time))
            event_frame_count += 1

            if co_occur:
                post_remaining = post_frames   # keep extending while both visible

            if event_frame_count >= max_frames:
                print(f"    → event capped at {fmt(video_time)}")
                close_event(video_time)
                event_active      = False
                event_frame_count = 0
                frames_since_event = 0
                post_remaining    = 0
        else:
            frames_since_event += 1

    # ── Wrap up ───────────────────────────────────────────────────────
    if event_active and writer is not None:
        close_event(video_time)

    cap.release()
    elapsed = time.time() - t_start

    # Write log
    log_path = clips_dir / "event_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Car+Bus Co-occurrence Log — {input_path}\n")
        f.write(f"Total events: {event_num}\n\n")
        for line in event_log:
            f.write(line + "\n")

    print(f"\n{'═'*60}")
    print(f"  Done in       : {fmt(elapsed)}")
    print(f"  Events found  : {event_num}")
    print(f"  YOLO calls    : {n_yolo:,} / {n_read:,} frames")
    print(f"  Clips in      : {clips_dir}/")
    print(f"{'═'*60}")
    if event_log:
        print("\n  ── Timeline ──")
        for line in event_log:
            print(f"    {line}")
    else:
        print("\n  No car+bus co-occurrence events found in this video.")
    print()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default=config.INPUT_VIDEO)
    p.add_argument("--out-folder", default=DEFAULT_OUT_FOLDER)
    args = p.parse_args()
    run(args.input, args.out_folder)
