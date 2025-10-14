## 1 Overview and Goals
The **Screen Time Analyzer** is a system designed to measure the on‑screen presence of cast members in episodic video content (e.g. reality TV shows).  Its goals are:

- **High accuracy and efficiency** – identify and measure every instance a cast member appears on screen, even if their face is partially occluded or turned away.  Accuracy should exceed 95% and manual effort should be minimal.
- **Scalable pipeline** – handle large video files, process each episode end‑to‑end and accumulate cast member appearances across multiple episodes and seasons.
- **Minimal user intervention** – once the system is configured, the user simply uploads a new episode.  The pipeline performs detection, tracking, recognition and summarization automatically.
- **Flexible exports** – produce CSV/JSON timelines, segment breakdowns, totals and video overlays.  Future iterations should support Google Sheets integration.
- **Iterative improvement** – allow manual review and correction of face assignments to improve the facebank over time.

## 2 Pipeline Overview
The end‑to‑end process comprises several stages.  Each stage corresponds to a script/module in the repository and can be run separately or through a master pipeline script.

1. **Episode Upload & Ingestion**
   - User places a new episode video (e.g. `s06e02.mp4`) into the `/videos` folder and updates `configs/episodes.yaml` with metadata (title, season, episode number and cast list).
   - The pipeline reads the episode and divides it into frames at the native frame rate (e.g. 23.976 fps).

2. **Detection & Tracking**
   - **Two cooperating signals (both required):**
     - **Face signal (accuracy):** RetinaFace (InsightFace) for detection + alignment to 112×112; ArcFace for embeddings and identity.
     - **Body continuity signal (credit when face turns):** YOLO (person class) + **ByteTrack**. The person tracker is the **authoritative continuity backbone**. Once a track is labeled by a face match, the identity **persists** while the tracker keeps the ID alive (configurable gap tolerance).
   - Detected frames and tracks are stored under `data/harvest/<episode>/` with a manifest containing bounding box coordinates, frame numbers and confidences.

3. **Recognition & Facebank Management**
   - **Embedding extraction**: For each harvested face crop, compute an embedding vector using the ArcFace model.
   - **Facebank**: Maintain a database of known cast members with one or more reference images per person.  A script (e.g. `build_facebank.py`) processes images stored under `/data/facebank/<name>/` to compute their embeddings.  It outputs `facebank.parquet` and metadata JSON.
   - **Matching**: For each track, compare its average embedding to the facebank.  If the cosine similarity exceeds a threshold (e.g. `0.4`), assign that track to the corresponding cast member.  Unknown faces are left unassigned.
   - **Improving the facebank**: When new faces are discovered or assignments are incorrect, the user can move crops into the appropriate person folder and rebuild the facebank.

4. **Harvest & Ground Truth Assignment**
   - **Harvest (bootstrap images from episode; mandatory):**
     1. Iterate frames (`stride=1` by default).
     2. Detect persons (YOLO) → **ByteTrack** to obtain stable `track_id`s per individual.
     3. Detect faces (RetinaFace). For each face:
        - **Align** to **112×112** using InsightFace landmarks (fallback to crop→resize if landmarks fail).
        - **Associate** faces to person tracks with a relaxed IoU gate (**≥ 0.20**) **after dilating** the track box by 15% on each side and looking back up to ±1 frame for lagging detections; fall back to a center-in-box match when exactly one face candidate lies inside the track.
     4. **Sampling:** keep **4–8** aligned crops per `track_id`, enforce `min_gap_frames` (**8**), gate on face size (`min_area_frac=0.005`), **frontalness ≥ 0.35**, and drop candidates below the track’s 40th percentile Laplacian sharpness. Quality scoring weights sharpness/frontality/area at **0.35/0.45/0.20**, with at least one high-frontal pick preserved.
     5. **Write:**
        - `data/harvest/<video_stem>/track_####/*.jpg` (aligned crops)
        - `data/harvest/<video_stem>/manifest.json` with per-track stats: `track_id`, `total_frames`, `avg_conf`, `avg_area`, `samples[]`, `first_ts_ms`, `last_ts_ms`.
   - **Manual assignment (optional)**: The manifest helps identify unassigned tracks.  Users can review sample frames and assign the track to a cast member by moving the crops into the correct folder in the facebank.  After manual assignment, rebuild the facebank.

5. **Attribution & Summarization**
   - **Associate tracks**: Merge overlapping face tracks and fill short gaps (e.g. gaps ≤ `max_gap_ms`) to get continuous appearance intervals.
   - **Calculate run time**: For each cast member, sum the durations of all their attributed intervals.  Ignore runs shorter than a configurable minimum (e.g. `min_run_ms`).
   - **Outputs**:
     - `*-events.json`: raw events with start and end times for each track.
     - `*-segments.csv`: list of merged segments per cast member with start/end timestamps (ms), frame numbers and durations.
     - `*-TOTALS.csv/json`: summary of total screen time per cast member for the episode.
     - `*-timeline.csv`: timeline view with second-by-second presence of each cast member (1/0 per second).

6. **QA & Export**
   - **Video overlay**: A visualization script overlays bounding boxes, track IDs, assigned labels and similarity scores onto the original episode and exports an MP4 for quality control.
   - **Review & corrections**: Users can watch the overlay video to spot mis-assignments and adjust the facebank.
   - **Export**: The system outputs CSV and JSON to `data/outputs/<episode>/`.  A separate script can upload results to Google Sheets or another data store in the future.

## 3 Directories & Key Files
**Repository layout (concrete & binding)**

/configs
  pipeline.yaml                # thresholds and knobs
  bytetrack.yaml               # tracker settings

/screentime
  /detectors
    face_retina.py             # RetinaFace wrapper (InsightFace)
    person_yolo.py             # YOLO person detector
  /tracking
    bytetrack_wrap.py
  /recognition
    embed_arcface.py
    facebank.py
    matcher.py                 # cosine + voting persistence
  /harvest/harvest.py
  /attribution/{associate.py,aggregate.py}
  /viz/overlay.py
  io_utils.py
  types.py

/scripts
  harvest_faces.py             # harvest flow (writes data/harvest/* + manifest)
  build_facebank.py            # build centroid embeddings → data/facebank.parquet
  run_tracker.py               # YOLO+ByteTrack + RetinaFace + vote persistence
  make_overlays.py             # render labeled overlay mp4
  export_totals.py             # CSV/JSON exports (totals, segments, timeline)

**requirements.txt (explicit)**

insightface>=0.7.3
onnxruntime>=1.20.0           # use onnxruntime-gpu if we add CUDA later
ultralytics>=8.3.0
opencv-python>=4.9.0
numpy>=1.26.0
pandas>=2.2.0
pyarrow>=15.0.0
tqdm>=4.66.0
pillow>=10.3.0
streamlit>=1.37.0             # optional labeler UI

(Add `.gitignore` for `data/`, `.venv/`, `models/weights/*`.)

## 4 Configuration Highlights
- **`pipeline.yaml`**
  - `stride`: number of frames to skip when running detection (smaller = more accurate but slower).
  - `det_conf`: minimum face detection confidence (e.g. 0.08 for YOLO).  Adjust to reduce false positives.
  - `min_area`: minimum face bounding box area to consider.
  - `max_gap_ms`: maximum gap between runs to bridge into a single appearance.
  - `min_run_ms`: minimum duration of a segment; discard runs shorter than this.
  - `similarity_threshold`: minimum cosine similarity for assigning a track to a cast member.

- **`bytetrack.yaml`**
  - Tracker parameters such as `track_thresh`, `high_thresh`, `match_thresh`, `nms_thresh`, `max_time_lost`.  Tune for stable tracking.

## 5 Implementation Notes
1. **Detection Model Choice**:
   - *RetinaFace* provides strong face detection accuracy.  YOLOv8n supplies robust person detections for body continuity, while RetinaFace via `insightface` feeds ArcFace.
   - The body continuity path (YOLO + ByteTrack) is **not optional**.  All screen-time attribution is computed over person tracks; face matches assign names to tracks and those labels persist through non-frontal frames.

2. **Embedding & Matching**:
   - Use the ArcFace model (`w600k_r50.onnx`) to compute 512‑dimensional embeddings.  Embeddings are averaged across all crops in a track before matching.
   - Build a robust facebank: include multiple reference images per cast member with different lighting, angles and expressions.  Rebuild the facebank whenever new high-quality images are available.

3. **Harvest & Manual Assignment**:
   - **Harvest (bootstrap images from episode; mandatory):**
     1. Iterate frames (`stride=1` by default).
     2. Detect persons (YOLO) → **ByteTrack** to obtain stable `track_id`s per individual.
     3. Detect faces (RetinaFace). For each face:
        - **Align** to **112×112** using InsightFace landmarks (fallback to crop→resize if landmarks fail).
        - **Associate** faces to person tracks with IoU **≥ 0.20** after 15% box dilation and ±1 frame tolerance; fall back to center-in-box when unique.
     4. **Sampling:** keep **4–8** representative aligned crops per `track_id`, enforce `min_gap_frames=8`, `min_area_frac=0.005`, and the frontalness/sharpness gates described above.
     5. **Write:**
        - `data/harvest/<video_stem>/track_####/*.jpg` (aligned crops)
        - `data/harvest/<video_stem>/manifest.json` with per-track stats: `track_id`, `total_frames`, `avg_conf`, `avg_area`, `samples[]`, `first_ts_ms`, `last_ts_ms`.
   - **Manual assignment (optional)**: Move representative crops of each new or misidentified face into `/data/facebank/<name>/` and rerun `build_facebank.py`.

4. **Quality Assurance**:
   - Render overlay videos to visually inspect track assignments.  Adjust thresholds if too many false positives or false negatives occur.
   - Use the timeline CSV to cross‑check totals across episodes.  Investigate large discrepancies.

## 6 Label Persistence & Defaults
**Label persistence & defaults**
- **similarity_th:** 0.82 (cosine). If a track collects votes for a label with top weight ≥ runner-up × 2, set `track_identity=label`. Apply **vote_decay** 0.99 per frame so stale votes fade.
- **flip_tolerance:** require ≥ 30% margin over current label’s cumulative weight to flip an identity on a live track.
- **segments:** bridge gaps ≤ **max_gap_ms=500**, drop micro-runs < **min_run_ms=200**.
- **face association:** IoU ≥ 0.20 with 15% box dilation and ±1 frame history; fallback to center-in-box when unique.
- **harvest defaults:** `samples_per_track=8`, `min_gap_frames=8`, `min_area_frac=0.005`, `min_frontalness=0.35`, `min_sharpness_pct=40`, `det_size=[960,960]`, `face_conf_th=0.45`, `person_conf_th=0.20`.

## 7 Future Work
- **Automation**: Provide a master script (e.g. `run_pipeline.py`) to orchestrate the entire flow given an episode.
- **Google Sheets Export**: Build a script to push totals and timelines to a Google Sheet via its API.
- **UI/Web App**: Develop a simple web or desktop app for uploading episodes, running the pipeline and viewing results.
