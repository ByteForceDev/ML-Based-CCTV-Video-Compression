# ─────────────────────────────────────────────────────────────
#  Configuration – edit these values before running the script
# ─────────────────────────────────────────────────────────────

# Root folder for raw input videos (kept out of Git).
DATA_FOLDER = "data"

# Root folder where all derived clip folders will be written.
OUTPUT_ROOT = "output"

INPUT_VIDEO = "input_video.mp4"

# Folder name (under OUTPUT_ROOT) where individual event clips will be saved.
# Each event gets its own file: event_001_00h01m26s_bus_car.mp4
CLIPS_FOLDER = "event_clips"

# ── Tier-1  Background Subtraction ───────────────────────────
# Minimum contour area (px²) to be considered real motion.
# Kept as an ABSOLUTE value — not scaled by resolution — so that
# background objects (small, far away) are also caught.
MOTION_AREA_THRESHOLD = 500
MIN_MOTION_FRAMES     = 1
MOG2_HISTORY          = 500
MOG2_VAR_THRESHOLD    = 16    # OpenCV default; lower = more sensitive
MOG2_DETECT_SHADOWS   = True

# ── Speed: frame sampling interval ───────────────────────────
# Only decode + analyse 1 out of every N frames while scanning
# (between events).  During active event clips every frame is
# decoded to keep clips smooth.  1 = off (process every frame).
# For a 30fps video, DETECTION_INTERVAL=3 gives ~10fps detection
# and roughly 3× throughput improvement without missing events.
DETECTION_INTERVAL = 2

# ── Tier-2  YOLOv8 + ByteTrack Object Tracker ────────────────
YOLO_MODEL      = "yolov8n.pt"
YOLO_DEVICE     = "cuda"         # 'cpu' if no GPU
YOLO_CONFIDENCE = 0.35           # 0.35 balances sensitivity vs false positives

# Minimum bounding-box area (px²) for a detection to be considered real.
# Filters out tiny corner / compression artefact detections.
MIN_BOX_AREA    = 1500

# A track must appear in this many consecutive sampled frames before it
# can trigger an event.  Prevents single-frame ghost detections from
# generating clips (false positives on corners, railings, etc.).
MIN_TRACK_AGE   = 2

# COCO class IDs to detect and trigger events for.
# Vehicles
VEHICLE_IDS  = {2, 3, 5, 7}            # car, motorcycle, bus, truck
# People & micro-mobility
PEOPLE_IDS   = {0, 1}                  # person, bicycle
# Animals  (any animal on a road/property is a notable event)
ANIMAL_IDS   = {14, 15, 16, 17, 18,   # bird, cat, dog, horse, sheep
                19, 20, 21, 22, 23}   # cow, elephant, bear, zebra, giraffe
# Suspicious / notable objects
OBJECT_IDS   = {24, 26, 28}           # backpack, handbag, suitcase

RELEVANT_CLASS_IDS = VEHICLE_IDS | PEOPLE_IDS | ANIMAL_IDS | OBJECT_IDS

# ── Event Detection ───────────────────────────────────────────
# A key event fires using heuristic triggering when a BRAND-NEW
# track ID (object) appears that has never been
# seen before in the entire video.
#
# PRE_ROLL_SECONDS  : seconds of footage BEFORE the trigger (context)
# POST_ROLL_SECONDS : seconds of footage AFTER the last new detection
# MAX_EVENT_SECONDS : hard cap – one event clip cannot exceed this
# MIN_IDLE_SECONDS  : minimum quiet gap required after an event ends
#                     before the next event can start

PRE_ROLL_SECONDS  = 1
POST_ROLL_SECONDS = 1
MAX_EVENT_SECONDS = 15
MIN_IDLE_SECONDS  = 3

# If True, keep extending event post-roll while relevant detections remain
# visible. Disable for compact clips focused on event snapshots.
EXTEND_EVENT_WHILE_VISIBLE = False

# In heuristic mode, also allow event trigger on sustained subject presence
# (not only on brand-new track IDs) to avoid missing short in-between activity.
HEURISTIC_TRIGGER_ON_PRESENCE = True
MIN_SUBJECT_FRAMES_TRIGGER    = 2

# ── Output Clips ────────────────────────────────────────────
OUTPUT_FPS   = 25       # baseline output FPS for saved clips
OUTPUT_CODEC = "h265"   # 'h265' (recommended) or 'h264'; auto-fallback

# Compression mode used by main pipeline.
# normal     : faster encode, good quality
# aggressive : smaller files, slower encode
COMPRESSION_MODE = "normal"

# H.265 profile tuning
H265_PRESET_NORMAL     = "fast"
H265_CRF_NORMAL        = 24
H265_PRESET_AGGRESSIVE = "slow"
H265_CRF_AGGRESSIVE    = 30

# H.264 fallback profile tuning
H264_PRESET_NORMAL     = "fast"
H264_CRF_NORMAL        = 23
H264_PRESET_AGGRESSIVE = "slow"
H264_CRF_AGGRESSIVE    = 28

# Playback speed-up for saved event clips. 1.0 keeps natural speed.
# Values >1.0 write clips at higher FPS (shorter playback duration).
EVENT_SPEEDUP_FACTOR = 2.5

# ── Low-FPS Boost ────────────────────────────────────────────
# When input video FPS is at or below this threshold the output
# clips are written at LOW_FPS_BOOST_TO fps instead.  This makes
# the clip play faster (all frames kept, just sped-up) so a
# 40-second clip at 4 fps becomes ~8 seconds at 20 fps.
# Set LOW_FPS_THRESHOLD = 0 to disable.
LOW_FPS_THRESHOLD = 10   # fps — trigger speed-up below this value
LOW_FPS_BOOST_TO  = 20   # fps — output playback rate for boosted clips

# ── Display ──────────────────────────────────────────────────
SHOW_PREVIEW = False
