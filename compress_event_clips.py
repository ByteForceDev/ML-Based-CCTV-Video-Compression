#!/usr/bin/env python3
"""
compress_event_clips.py
=======================
Post-compress already extracted event clips into a new folder.

This script does NOT run detection and does NOT modify main.py.
It is intended to be fast and safe for iterative size optimization.

Examples:
    python compress_event_clips.py
    python compress_event_clips.py --input-folder main_ip_clips_high_recall --output-folder main_ip_clips_high_recall_compact
    python compress_event_clips.py --encoder nvenc --cq 30 --preset p5
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ffmpeg_has_encoder(ffmpeg_bin: str, name: str) -> bool:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in proc.stdout


def _build_encode_cmd(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    encoder: str,
    cq: int,
    preset: str,
) -> list[str]:
    base = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]

    if encoder == "hevc_nvenc":
        codec = [
            "-c:v", "hevc_nvenc",
            "-preset", preset,
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]
    elif encoder == "av1_nvenc":
        codec = [
            "-c:v", "av1_nvenc",
            "-preset", preset,
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]
    elif encoder == "libx265":
        codec = [
            "-c:v", "libx265",
            "-preset", "fast",
            "-crf", str(cq),
            "-tag:v", "hvc1",
        ]
    else:
        codec = [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(cq),
        ]

    tail = ["-movflags", "+faststart", str(output_path)]
    return base + codec + tail


def _folder_size_bytes(folder: Path) -> int:
    total = 0
    for p in folder.glob("*.mp4"):
        total += p.stat().st_size
    return total


def _update_event_log_sizes(log_path: Path, out_clips_bytes: int) -> None:
    """Rewrite the Size Reduction block in an existing event_log.txt."""
    if not log_path.exists():
        return

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    input_bytes = None
    frames_kept_line_idx = None
    size_block_start = None

    for i, line in enumerate(lines):
        if "Input video" in line:
            m = re.search(r"\(([0-9,]+) bytes\)", line)
            if m:
                input_bytes = int(m.group(1).replace(",", ""))
        if "Frames kept" in line:
            frames_kept_line_idx = i
        if line.startswith("── Size Reduction"):
            size_block_start = i

    if input_bytes is None or size_block_start is None:
        return

    saved_bytes = input_bytes - out_clips_bytes
    reduction_pct = 100.0 * saved_bytes / max(input_bytes, 1)

    input_mb = input_bytes / 1_048_576
    clips_mb = out_clips_bytes / 1_048_576
    saved_mb = saved_bytes / 1_048_576

    lines[size_block_start + 1] = (
        f"  Input video    :    {input_mb:,.1f} MB  ({input_bytes:,} bytes)"
    )
    lines[size_block_start + 2] = (
        f"  All clips      :    {clips_mb:,.1f} MB  ({out_clips_bytes:,} bytes)"
    )
    lines[size_block_start + 3] = f"  Space saved    :    {saved_mb:,.1f} MB"
    lines[size_block_start + 4] = f"  Size reduction : {reduction_pct:.1f}%"

    # Keep Frames kept line unchanged if present; it belongs to detection pass.
    if frames_kept_line_idx is not None:
        pass

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast post-compression for event clips")
    ap.add_argument("--input-folder", default="main_ip_clips_high_recall")
    ap.add_argument("--output-folder", default="main_ip_clips_high_recall_compact")
    ap.add_argument(
        "--encoder",
        default="auto",
        choices=["auto", "nvenc", "av1_nvenc", "x265", "x264"],
        help="Encoder family. 'auto' tries hevc_nvenc, then av1_nvenc, then libx265.",
    )
    ap.add_argument("--cq", type=int, default=30, help="Quality target. Higher = smaller files.")
    ap.add_argument("--preset", default="p5", help="NVENC preset p1..p7 (p7 slowest/best).")
    ap.add_argument(
        "--update-log-only",
        action="store_true",
        help="Only refresh size stats in output-folder event_log.txt using current .mp4 files.",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)

    if not in_dir.exists():
        sys.exit(f"[ERROR] Input folder not found: {in_dir}")

    clips = sorted(p for p in in_dir.glob("*.mp4") if p.is_file())
    if not clips:
        sys.exit(f"[ERROR] No .mp4 clips found in: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    if args.update_log_only:
        out_bytes = _folder_size_bytes(out_dir)
        log_path = out_dir / "event_log.txt"
        if not log_path.exists():
            # Fallback: seed log from source then rewrite numbers.
            src_log = in_dir / "event_log.txt"
            if src_log.exists():
                shutil.copy2(src_log, log_path)
        _update_event_log_sizes(log_path, out_bytes)
        print("\n" + "=" * 62)
        print(f"  Updated log    : {log_path}")
        print(f"  Output clips   : {out_bytes / 1_048_576:.1f} MB")
        print("=" * 62)
        return

    # Select concrete encoder.
    if args.encoder == "auto":
        if _ffmpeg_has_encoder(ffmpeg_bin, "hevc_nvenc"):
            encoder = "hevc_nvenc"
        elif _ffmpeg_has_encoder(ffmpeg_bin, "av1_nvenc"):
            encoder = "av1_nvenc"
        else:
            encoder = "libx265"
    elif args.encoder == "nvenc":
        encoder = "hevc_nvenc"
    elif args.encoder == "av1_nvenc":
        encoder = "av1_nvenc"
    elif args.encoder == "x265":
        encoder = "libx265"
    else:
        encoder = "libx264"

    print("\n" + "-" * 62)
    print(f"  Source folder : {in_dir}")
    print(f"  Output folder : {out_dir}")
    print(f"  Clips found   : {len(clips)}")
    print(f"  Encoder       : {encoder}")
    print(f"  CQ/CRF        : {args.cq}")
    print("-" * 62 + "\n")

    t0 = time.time()
    ok_count = 0
    fail_count = 0

    for i, src in enumerate(clips, 1):
        dst = out_dir / src.name
        cmd = _build_encode_cmd(
            ffmpeg_bin=ffmpeg_bin,
            input_path=src,
            output_path=dst,
            encoder=encoder,
            cq=args.cq,
            preset=args.preset,
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            ok_count += 1
        except subprocess.CalledProcessError:
            fail_count += 1
            print(f"[WARN] Failed: {src.name}")
            continue

        if i % 10 == 0 or i == len(clips):
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(clips)}]  elapsed={_fmt(elapsed)}")

    # Copy event log if present
    event_log = in_dir / "event_log.txt"
    if event_log.exists():
        shutil.copy2(event_log, out_dir / "event_log.txt")

    elapsed = time.time() - t0

    in_bytes = _folder_size_bytes(in_dir)
    out_bytes = _folder_size_bytes(out_dir)
    _update_event_log_sizes(out_dir / "event_log.txt", out_bytes)
    saved = in_bytes - out_bytes
    pct = 100.0 * saved / max(in_bytes, 1)

    print("\n" + "=" * 62)
    print(f"  Completed      : {_fmt(elapsed)}")
    print(f"  Success/Fail   : {ok_count}/{fail_count}")
    print(f"  Input clips    : {in_bytes / 1_048_576:.1f} MB")
    print(f"  Output clips   : {out_bytes / 1_048_576:.1f} MB")
    print(f"  Saved          : {saved / 1_048_576:.1f} MB  ({pct:.1f}%)")
    print("=" * 62)


if __name__ == "__main__":
    main()
