#!/usr/bin/env python3
"""
main.py — CCTV Intelligent Event Summariser
============================================
3-Stage Processing Pipeline:

  Stage 1 — Background Subtraction (MOG2)
            Fast motion gate.  Discards static frames before any
            neural network is invoked.  Only frames with contour area
            above the threshold pass through to Stage 2.

  Stage 2 — Object Detection + Tracking (YOLOv8-nano + ByteTrack)
            Identifies and tracks every relevant COCO object class.
            Only runs on frames that cleared Stage 1.

    Stage 3 — Clip Encoding (FFmpeg H.265 / H.264)
            Pre-roll + post-roll event clips are encoded with
            -movflags faststart for efficient storage and instant
            seeking.

Usage:
    python main.py
    python main.py --input cctv_prison.mp4 --out-folder prison_clips
    python main.py --input cctv_prison.mp4 --out-folder prison_clips --device cpu
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

# Explicitly load the CCTV config.py from the project root.
# This is required because the portfolio project left behind a config/
# package that would otherwise shadow the root config.py.
_cfg_spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).parent / "config.py"
)
config = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(config)
del _cfg_spec

# ── COCO class names (surveillance-relevant subset) ───────────────────
COCO_NAMES: dict = {
    0:  "person",    1:  "bicycle",   2:  "car",       3:  "motorcycle",
    5:  "bus",       7:  "truck",
    14: "bird",      15: "cat",       16: "dog",       17: "horse",
    18: "sheep",     19: "cow",       20: "elephant",  21: "bear",
    22: "zebra",     23: "giraffe",
    24: "backpack",  26: "handbag",   28: "suitcase",
}

# Detection box colours (BGR) per category
_CLR_PERSON  = (0,   220, 255)   # yellow
_CLR_VEHICLE = (0,   180, 255)   # sky-blue
_CLR_ANIMAL  = (0,   255, 120)   # green
_CLR_OBJECT  = (200, 100, 255)   # purple


def _box_color(cls_id: int) -> tuple:
    if cls_id in config.PEOPLE_IDS:   return _CLR_PERSON
    if cls_id in config.VEHICLE_IDS:  return _CLR_VEHICLE
    if cls_id in config.ANIMAL_IDS:   return _CLR_ANIMAL
    return _CLR_OBJECT


# ── Helpers ───────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    """Format a duration in seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resolve_input_path(path: str) -> str:
    """Resolve an input video path, preferring DATA_FOLDER when present.

    Order of precedence:
      1) Absolute path if it exists
      2) Relative path from project root if it exists
      3) DATA_FOLDER / relative path if it exists
      4) Original string (so OpenCV can still report a clear error)
    """
    p = Path(path)
    # Absolute path
    if p.is_absolute() and p.exists():
        return str(p)

    # Relative from project root
    if p.exists():
        return str(p)

    data_root = getattr(config, "DATA_FOLDER", None)
    if data_root:
        candidate = Path(data_root) / p
        if candidate.exists():
            return str(candidate)

    return str(p)


def _resolve_output_folder(folder: str) -> str:
    """Resolve an output folder under OUTPUT_ROOT when configured.

    - Absolute paths are returned unchanged.
    - If OUTPUT_ROOT is set and the given folder is not already under it,
      the folder is treated as a subdirectory of OUTPUT_ROOT.
    """
    p = Path(folder)
    if p.is_absolute():
        return str(p)

    output_root = getattr(config, "OUTPUT_ROOT", None)
    if not output_root:
        return str(p)

    root = Path(output_root)
    # If already rooted at OUTPUT_ROOT, leave as-is
    try:
        p.relative_to(root)
        return str(p)
    except ValueError:
        return str(root / p)


def _clip_filename(event_num: int, video_time: float,
                   class_names: list) -> str:
    """
    Build a clip filename such as:
        event_003__00h08m42s__person.mp4
        event_001__00h01m26s__bus_truck.mp4
    class_names: list of class name strings for the triggering tracks
                 (duplicates allowed — they are deduped and sorted).
    """
    h = int(video_time // 3600)
    m = int((video_time % 3600) // 60)
    s = int(video_time % 60)
    label = "_".join(sorted(set(class_names)))[:40]
    return f"event_{event_num:03d}__{h:02d}h{m:02d}m{s:02d}s__{label}.mp4"


def _open_writer(path: str, width: int, height: int,
                 fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def _encode(
    raw_path: str,
    final_path: str,
    ffmpeg_bin: str,
    compression_mode: str,
) -> None:
    """Re-encode raw mp4v clip using selectable compression mode."""
    aggressive = compression_mode == "aggressive"

    h265_preset = (config.H265_PRESET_AGGRESSIVE
                   if aggressive else config.H265_PRESET_NORMAL)
    h265_crf = (config.H265_CRF_AGGRESSIVE
                if aggressive else config.H265_CRF_NORMAL)
    h264_preset = (config.H264_PRESET_AGGRESSIVE
                   if aggressive else config.H264_PRESET_NORMAL)
    h264_crf = (config.H264_CRF_AGGRESSIVE
                if aggressive else config.H264_CRF_NORMAL)

    h265_args = [
        "-c:v", "libx265",
        "-preset", h265_preset,
        "-crf", str(h265_crf),
        "-tag:v", "hvc1",
    ]
    h264_args = [
        "-c:v", "libx264",
        "-preset", h264_preset,
        "-crf", str(h264_crf),
    ]

    codec_args = h265_args if config.OUTPUT_CODEC == "h265" else h264_args
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw_path]
            + codec_args
            + ["-movflags", "+faststart", final_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Silently fall back to H.264
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", raw_path]
            + h264_args
            + ["-movflags", "+faststart", final_path],
            check=True, capture_output=True,
        )
    Path(raw_path).unlink(missing_ok=True)


def _annotate(frame: np.ndarray, boxes, video_time: float,
              event_num: int) -> np.ndarray:
    """Draw bounding boxes + top status bar on a copy of the frame."""
    out = frame.copy()
    w   = out.shape[1]
    for box in boxes:
        cls_id          = int(box.cls[0])
        conf            = float(box.conf[0])
        tid             = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color           = _box_color(cls_id)
        name            = COCO_NAMES.get(cls_id, f"cls{cls_id}")
        label           = (f"{name}#{tid}  {conf:.0%}"
                           if tid >= 0 else f"{name}  {conf:.0%}")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Status bar
    cv2.rectangle(out, (0, 0), (w, 26), (0, 50, 0), -1)
    cv2.putText(out, f"EVENT #{event_num:03d}  {_fmt(video_time)}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return out


# ── Main pipeline ─────────────────────────────────────────────────────

def run(
    input_path: str,
    out_folder: str,
    device: str,
    compression_mode: str = "normal",
    max_percent: float = 100.0,
    progress_callback = None,
) -> None:
    # Resolve logical locations into the configured data/output roots.
    input_path = _resolve_input_path(input_path)
    out_folder = _resolve_output_folder(out_folder)

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional test mode: process only a percentage of the input video.
    pct = max(0.1, min(100.0, float(max_percent)))
    target_frames = int(total_frames * (pct / 100.0))
    target_frames = max(1, min(total_frames, target_frames))

    # Motion threshold used as-is (absolute px²).
    # Do NOT scale upward for larger resolutions — that would filter out
    # small background objects and miss far-away activity.
    motion_thr = config.MOTION_AREA_THRESHOLD

    pre_frames  = int(config.PRE_ROLL_SECONDS  * video_fps)
    post_frames = int(config.POST_ROLL_SECONDS * video_fps)
    max_frames  = int(config.MAX_EVENT_SECONDS * video_fps)
    gap_frames  = int(config.MIN_IDLE_SECONDS  * video_fps)

    # ── Clip playback rate: never slower than source, optional speed-up ─
    base_clip_fps = max(video_fps, float(config.OUTPUT_FPS))
    speedup_mul   = max(1.0, float(getattr(config, "EVENT_SPEEDUP_FACTOR", 1.0)))
    clip_write_fps = min(60.0, base_clip_fps * speedup_mul)
    speedup_factor = clip_write_fps / video_fps

    # ── Low-FPS boost: keep existing behavior for very low-fps sources ─
    low_fps_boost = (
        config.LOW_FPS_THRESHOLD > 0
        and video_fps <= config.LOW_FPS_THRESHOLD
    )
    if low_fps_boost:
        clip_write_fps = max(clip_write_fps, float(config.LOW_FPS_BOOST_TO))
        speedup_factor = clip_write_fps / video_fps

    # For very low-fps inputs every frame is precious; don't skip any.
    effective_interval = 1 if low_fps_boost else config.DETECTION_INTERVAL

    clips_dir  = Path(out_folder)
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    # ── Print header ──────────────────────────────────────────────────
    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  Input       : {input_path}")
    print(f"  Clips saved : {clips_dir}/")
    print(f"  Video       : {width}×{height}  {video_fps:.1f}fps  "
          f"{_fmt(total_frames / video_fps)}")
    if target_frames < total_frames:
        print(f"  Test mode   : {pct:.1f}%  ({target_frames:,}/{total_frames:,} frames)")
    print(f"  Pre/Post    : {config.PRE_ROLL_SECONDS}s / {config.POST_ROLL_SECONDS}s"
          f"    Max clip: {config.MAX_EVENT_SECONDS}s    Gap: {config.MIN_IDLE_SECONDS}s")
    print(f"  Motion thr  : {motion_thr} px²  (absolute, no resolution scaling)")
    print(f"  Compression : {compression_mode}")
    if speedup_factor > 1.01:
        print(f"  FPS boost   : {video_fps:.1f}fps → {clip_write_fps:.0f}fps output  "
              f"({speedup_factor:.1f}× speed-up, clips {speedup_factor:.1f}× shorter)")
    if effective_interval > 1:
        print(f"  Frame skip  : 1 in {effective_interval} frames decoded while scanning  "
              f"(∼{video_fps/effective_interval:.0f}fps detection)")
    print(SEP)

    # ── Stage 2: YOLO model ───────────────────────────────────────────
    print("\n[Stage-2] Loading YOLOv8 model …")
    model = YOLO(config.YOLO_MODEL)
    model.to(device)
    print(f"[Stage-2] Model ready on {device}\n")
    if progress_callback: progress_callback({'log': '[Stage-2] YOLOv8 Model Loaded', 'progress': 0, 'events': 0})

    print()  # blank line before first progress / event

    # ── Stage 1: Background subtractor ────────────────────────────────
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )

    # ── Pipeline state ─────────────────────────────────────────────────
    pre_buf: collections.deque = collections.deque(maxlen=pre_frames)
    seen_track_ids: set        = set()    # all track IDs ever seen

    event_active        = False
    event_num           = 0
    clips_written       = 0
    event_frame_count   = 0
    post_remaining      = 0
    frames_since_event  = gap_frames   # allow immediate event at start
    motion_run          = 0
    frames_saved_total  = 0

    writer     = None
    raw_path   = ""
    final_path = ""

    frame_n    = 0   # total grabbed frames (for accurate time & progress)
    n_read     = 0   # alias updated per-frame; kept for display compat
    n_yolo     = 0
    subject_run = 0

    # track_age: how many sampled frames each track ID has been seen.
    # A track only triggers an event once it reaches MIN_TRACK_AGE.
    track_age: dict = {}

    event_log: list = []
    t_start = time.time()

    # ── Helper: close and encode the active event clip ─────────────────
    def close_event(vt: float) -> None:
        nonlocal writer, clips_written
        if writer is None:
            return
        writer.release()
        writer = None
        try:
            _encode(raw_path, final_path, ffmpeg_bin, compression_mode)
            size_mb = Path(final_path).stat().st_size / 1_048_576
            print(f"    → clip saved:  {Path(final_path).name}  ({size_mb:.1f} MB)")
            clips_written += 1
        except subprocess.CalledProcessError as exc:
            print(f"    [WARN] encode failed: {exc}")
    # ── Frame loop ─────────────────────────────────────────────────────
    while frame_n < target_frames:
        grabbed = cap.grab()
        if not grabbed:
            break

        frame_n += 1
        n_read   = frame_n   # kept for display / summary compat

        # Progress line every 30 grabbed frames (frequent updates for UI)
        if frame_n % 30 == 0:
            elapsed = time.time() - t_start
            fps_p   = frame_n / max(elapsed, 1e-9)
            eta     = (target_frames - frame_n) / max(fps_p, 1e-9)
            pct_done = 100.0 * frame_n / max(target_frames, 1)
            msg = (f"  [{pct_done:5.1f}%]  {frame_n:,}/{target_frames:,}  "
                   f"{fps_p:.0f}fps  ETA {_fmt(eta)}  "
                   f"events={event_num}  clips_written={clips_written}")
            print(msg)
            if progress_callback:
                progress_callback({
                    'progress': pct_done,
                    'events': event_num,
                    'log': msg
                })


        # Between events: skip decoding non-sampled frames for speed.
        # During active events: always decode so clips are smooth.
        if not event_active and frame_n % effective_interval != 0:
            frames_since_event += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            continue

        video_time = frame_n / video_fps

        # ── Stage 1: Motion gate ───────────────────────────────────────
        fgmask  = subtractor.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        has_motion = any(cv2.contourArea(c) >= motion_thr for c in contours)

        if has_motion:
            motion_run += 1
        else:
            motion_run  = 0

        tier1 = motion_run >= config.MIN_MOTION_FRAMES
        pre_buf.append(frame.copy())

        if not tier1:
            # Static frame — handle post-roll for any active event
            if event_active:
                if post_remaining > 0:
                    writer.write(frame)
                    frames_saved_total += 1
                    post_remaining -= 1
                if post_remaining <= 0:
                    print(f"    → event ended at {_fmt(video_time)}")
                    close_event(video_time)
                    event_active       = False
                    event_frame_count  = 0
                    frames_since_event = 0
            else:
                frames_since_event += 1

            if config.SHOW_PREVIEW:
                cv2.imshow("CCTV Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ── Stage 2: YOLO + ByteTrack ──────────────────────────────────
        results = model.track(
            source=frame,
            conf=config.YOLO_CONFIDENCE,
            device=device,
            classes=list(config.RELEVANT_CLASS_IDS),
            persist=True,
            verbose=False,
        )
        n_yolo += 1

        new_tracks: list[str] = []   # class names for confirmed new tracks
        relevant_boxes = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id      = int(box.cls[0])
                tid         = int(box.id[0]) if box.id is not None else -1
                x1,y1,x2,y2= map(int, box.xyxy[0])
                box_area    = (x2 - x1) * (y2 - y1)

                # Skip detections whose bounding box is too small —
                # these are typically corner / railing artefacts.
                if box_area < config.MIN_BOX_AREA:
                    continue

                relevant_boxes.append(box)
                name = COCO_NAMES.get(cls_id, f"cls{cls_id}")

                if tid >= 0 and tid not in seen_track_ids:
                    # Increment age counter; only promote to a confirmed
                    # track once it has been seen MIN_TRACK_AGE times.
                    track_age[tid] = track_age.get(tid, 0) + 1
                    if track_age[tid] >= config.MIN_TRACK_AGE:
                        seen_track_ids.add(tid)
                        new_tracks.append(name)

        # Count consecutive frames with relevant detections.
        if relevant_boxes:
            subject_run += 1
        else:
            subject_run = 0

        # ── Trigger logic: high-recall heuristic (track + presence) ────
        in_gap = (not event_active) and (frames_since_event < gap_frames)
        trigger             = False
        trigger_new_tracks: list[str] = []

        if not event_active and not in_gap:
            # Heuristic: fire when a brand-new track ID appears.
            if new_tracks:
                trigger = True
                trigger_new_tracks = new_tracks
            # High-recall fallback: trigger on sustained subject presence
            # after the idle gap, even if track IDs are not new.
            elif (
                getattr(config, "HEURISTIC_TRIGGER_ON_PRESENCE", False)
                and subject_run >= int(getattr(config, "MIN_SUBJECT_FRAMES_TRIGGER", 2))
                and relevant_boxes
            ):
                trigger = True
                trigger_new_tracks = [
                    COCO_NAMES.get(int(b.cls[0]), "?")
                    for b in relevant_boxes[:4]
                ]

        # ── Open a new event clip ───────────────────────────────────────
        if trigger:
            event_num        += 1
            event_active      = True
            event_frame_count = 0
            post_remaining    = post_frames

            # Human-readable label: "person x2, car"
            cnt = collections.Counter(trigger_new_tracks)
            label_parts = [
                (f"{nm} x{n}" if n > 1 else nm)
                for nm, n in cnt.items()
            ]
            label_str = ", ".join(label_parts[:6]) if label_parts else "unknown"
            print(f"\n  ★ EVENT #{event_num} at {_fmt(video_time)}  ←  {label_str}")
            event_log.append(
                f"EVENT #{event_num:03d}  {_fmt(video_time)}  {label_str}"
            )

            # Clip filename uses unique sorted class names
            cls_names_for_file = trigger_new_tracks
            fname      = _clip_filename(event_num, video_time, cls_names_for_file)
            final_path = str(clips_dir / fname)
            raw_path   = final_path + ".raw.mp4"
            writer     = _open_writer(raw_path, width, height, clip_write_fps)

            # Write pre-roll buffer into the clip
            for buf_frame in pre_buf:
                writer.write(buf_frame)
                frames_saved_total += 1
            pre_buf.clear()

        # ── Write current frame into the active event clip ──────────────
        if event_active:
            writer.write(_annotate(frame, relevant_boxes, video_time, event_num))
            frames_saved_total += 1
            event_frame_count  += 1

            # Optional extension while object remains visible.
            if config.EXTEND_EVENT_WHILE_VISIBLE and relevant_boxes:
                post_remaining = post_frames

            # Hard cap — prevent runaway clips
            if event_frame_count >= max_frames:
                print(f"    → event capped at {_fmt(video_time)}")
                close_event(video_time)
                event_active       = False
                event_frame_count  = 0
                frames_since_event = 0
                post_remaining     = 0
        else:
            frames_since_event += 1

        if config.SHOW_PREVIEW:
            disp = (_annotate(frame, relevant_boxes, video_time, event_num)
                    if event_active else frame)
            cv2.imshow("CCTV Pipeline", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── End of video: close any still-open event ───────────────────────
    if event_active and writer is not None:
        print(f"    → event ended at {_fmt(video_time)}")
        close_event(video_time)

    cap.release()
    if config.SHOW_PREVIEW:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # ── Size reduction calculation ──────────────────────────────────────
    input_bytes = Path(input_path).stat().st_size
    clips_bytes = sum(
        f.stat().st_size for f in clips_dir.glob("*.mp4")
    )
    input_mb    = input_bytes / 1_048_576
    clips_mb    = clips_bytes / 1_048_576
    saved_mb    = input_mb - clips_mb
    reduction_pct = 100.0 * (1.0 - clips_bytes / max(input_bytes, 1))
    frames_pct_kept = 100.0 * frames_saved_total / max(n_read, 1)

    # ── Write event log file ────────────────────────────────────────────
    log_path = clips_dir / "event_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"CCTV Event Log — {input_path}\n")
        extracted_seconds = frames_saved_total / max(video_fps, 1)
        fh.write(f"Processed  : {_fmt(target_frames / video_fps)}"
                 f"  ({target_frames:,} frames @ {video_fps:.1f} fps)\n")
        fh.write(f"Extracted  : {_fmt(extracted_seconds)}\n")
        fh.write(f"Events     : {event_num}\n")
        fh.write("Stage-3    : heuristic trigger\n")
        if target_frames < total_frames:
            fh.write(f"Mode       : test ({pct:.1f}% of input)\n")
        if low_fps_boost:
            fh.write(f"FPS boost  : {video_fps:.1f}fps → {clip_write_fps:.0f}fps  "
                     f"({speedup_factor:.1f}x speed-up, clips {speedup_factor:.1f}x shorter)\n")
        fh.write("\n")
        fh.write("── Size Reduction ─────────────────────────────────\n")
        fh.write(f"  Input video    : {input_mb:>8.1f} MB  ({input_bytes:,} bytes)\n")
        fh.write(f"  All clips      : {clips_mb:>8.1f} MB  ({clips_bytes:,} bytes)\n")
        fh.write(f"  Space saved    : {saved_mb:>8.1f} MB\n")
        fh.write(f"  Size reduction : {reduction_pct:.1f}%\n")
        fh.write(f"  Frames kept    : {frames_saved_total:,} / {n_read:,}"
                 f"  ({frames_pct_kept:.1f}%)\n\n")
        fh.write("── Event Timeline ──────────────────────────────────\n")
        for entry in event_log:
            fh.write(entry + "\n")

    # ── Final summary (console) ─────────────────────────────────────────
    DSEP     = "═" * 62
    yolo_pct = 100.0 * n_yolo / max(n_read, 1)

    print(f"\n{DSEP}")
    print(f"  Completed in       : {_fmt(elapsed)}")
    print(f"  Frames processed   : {n_read:,} / {target_frames:,}")
    print(f"  YOLO calls (S2)    : {n_yolo:,}  ({yolo_pct:.1f}% of frames)")
    print("  Stage-3 mode       : heuristic trigger")
    print(f"  Compression mode   : {compression_mode}")
    print(f"  Key events found   : {event_num}")
    print(f"  Frames kept        : {frames_saved_total:,} / {n_read:,}  ({frames_pct_kept:.1f}%)")
    if speedup_factor > 1.01:
        print(f"  FPS boost          : {video_fps:.1f}fps → {clip_write_fps:.0f}fps  ({speedup_factor:.1f}x speed-up)")
    print(f"  Output folder      : {clips_dir}/")
    print(f"  Event log          : {log_path}")
    print(f"{'─'*62}")
    print(f"  ★ SIZE REDUCTION SUMMARY")
    print(f"    Input video  : {input_mb:>8.1f} MB")
    print(f"    All clips    : {clips_mb:>8.1f} MB")
    print(f"    Space saved  : {saved_mb:>8.1f} MB  →  {reduction_pct:.1f}% reduction")
    print(DSEP)

    if event_log:
        print("\n  ── Event Timeline ──")
        for entry in event_log:
            print(f"    {entry}")
    else:
        print("\n  No key events detected in this video.")
    print()

# ── Entry point ─────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CCTV Intelligent Event Summariser — 3-stage pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --input cctv_prison.mp4 --out-folder prison_clips
  python main.py --input cctv_prison.mp4 --out-folder prison_clips --device cpu
    python main.py --input main_ip.mp4 --out-folder main_ip_clips --compression-mode normal
    python main.py --input main_ip.mp4 --out-folder main_ip_clips_final --compression-mode aggressive
        """,
    )
    ap.add_argument(
        "--input",
        default=config.INPUT_VIDEO,
        help=f"Input video file  (default: {config.INPUT_VIDEO})",
    )
    ap.add_argument(
        "--out-folder",
        default=config.CLIPS_FOLDER,
        help=f"Output clips folder  (default: {config.CLIPS_FOLDER})",
    )
    ap.add_argument(
        "--device",
        default=config.YOLO_DEVICE,
        help=f"Compute device: cuda / cpu  (default: {config.YOLO_DEVICE})",
    )
    ap.add_argument(
        "--compression-mode",
        default=config.COMPRESSION_MODE,
        choices=["normal", "aggressive"],
        help=(
            "Compression profile for event clips: "
            "normal (faster, larger files) or aggressive (slower, smaller files)."
        ),
    )
    ap.add_argument(
        "--max-percent",
        type=float,
        default=100.0,
        help="Process only this percentage of the input video (for quick tests).",
    )
    args = ap.parse_args()
    run(
        args.input,
        args.out_folder,
        args.device,
        compression_mode=args.compression_mode,
        max_percent=args.max_percent,
    )


if __name__ == "__main__":
    main()