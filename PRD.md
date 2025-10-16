## 1 Overview and Goals
The **Screen Time Analyzer** is a system designed to measure the on‑screen presence of cast members in episodic video content (e.g. reality TV shows). Its goals are:

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
   - **Face detection**: Use *RetinaFace* (or YOLO-based face detector) to detect faces every N frames (e.g. stride 1 = every frame).  The detector outputs bounding boxes, confidence scores and landmarks.
   - **Person detection** (optional): Use a YOLO person detector if body tracking is desired when faces turn away.
   - **Multi-object tracking**: Run ByteTrack or another multi‑object tracker to link detections across frames into continuous tracks with unique IDs.  Tracks record the bounding box positions, start/end timestamps and detection confidence.
   - Detected frames and tracks are stored under `data/harvest/<episode>/` with a manifest containing bounding box coordinates, frame numbers and confidences.

3. **Recognition & Facebank Management**
   - **Embedding extraction**: For each harvested face crop, compute an embedding vector using the ArcFace model.
   - **Facebank**: Maintain a database of known cast members with one or more reference images per person.  A script (e.g. `build_facebank.py`) processes images stored under `/data/facebank/<name>/` to compute their embeddings.  It outputs `facebank.parquet` and metadata JSON.
   - **Matching**: For each track, compare its average embedding to the facebank.  If the cosine similarity exceeds a threshold (e.g. `0.4`), assign that track to the corresponding cast member.  Unknown faces are left unassigned.
   - **Improving the facebank**: When new faces are discovered or assignments are incorrect, the user can move crops into the appropriate person folder and rebuild the facebank.

4. **Harvest & Ground Truth Assignment**
   - **Harvest script (`harvest_faces.py`)**: Runs detection and tracking on an episode and saves face crops to `data/harvest/<episode>/tracks`.  It also generates `manifest.json` summarizing the tracks, frame ranges, average area and average confidence.
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
```
SCREEN TIME ANALYZER/
├── configs/
│   ├── pipeline.yaml         # thresholds (confidence, similarity), stride, min/max durations
│   └── bytetrack.yaml        # tracker parameters
├── scripts/                  # wrappers or orchestration scripts
├── app/                      # command-line or UI application (future)
├── data/
│   ├── videos/               # input episodes
│   ├── harvest/              # harvested face crops and manifests
│   ├── facebank/             # reference images and computed embeddings
│   └── outputs/              # CSV/JSON totals, segments, timeline, overlays
├── models/
│   └── weights/              # detector and recognition model weights (yolov8n_face.pt, arcface.onnx, etc.)
└── screentime/
    ├── detectors/
    │   ├── face_retina.py    # wrapper around RetinaFace or YOLO face detector
    │   └── person_yolo.py    # optional person detector
    ├── tracking/
    │   └── bytetrack_wrap.py # multi-object tracking wrapper
    ├── recognition/
    │   ├── embed_arcface.py  # compute embeddings from crops
    │   ├── facebank.py       # build and manage the facebank
    │   └── matcher.py        # match embeddings to facebank
    ├── harvest/
    │   └── harvest.py        # pipeline for detection, tracking and cropping
    ├── attribution/
    │   ├── associate.py      # merge tracks, fill gaps, remove short runs
    │   └── aggregate.py      # summarise events and totals
    ├── viz/
    │   └── overlay.py        # render overlay video for QA
    ├── io_utils.py           # common I/O helpers
    └── types.py              # common dataclasses and type hints
```

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
   - *RetinaFace* provides strong face detection accuracy.  YOLOv8n can be faster but may underperform on occluded faces.  Our pipeline uses RetinaFace via `insightface` with optional YOLO fallback.
   - Person detection is optional and should be used if bodies need to be tracked when faces turn away.  We can start with face-only detection and evaluate the need for body tracking.

2. **Embedding & Matching**:
   - Use the ArcFace model (`w600k_r50.onnx`) to compute 512‑dimensional embeddings.  Embeddings are averaged across all crops in a track before matching.
   - Build a robust facebank: include multiple reference images per cast member with different lighting, angles and expressions.  Rebuild the facebank whenever new high-quality images are available.

3. **Harvest & Manual Assignment**:
   - Use `harvest_faces.py` to generate tracks and sample frames.  Inspect samples to ensure the detections are correct.
   - Move representative crops of each new or misidentified face into `/data/facebank/<name>/` and rerun `build_facebank.py`.

4. **Quality Assurance**:
   - Render overlay videos to visually inspect track assignments.  Adjust thresholds if too many false positives or false negatives occur.
   - Use the timeline CSV to cross‑check totals across episodes.  Investigate large discrepancies.

## 6 Future Work & MCP Considerations
- **VS Code MCP Integration**: The VS Code MCP server allows your assistant to request code diagnostics and open files, but due to security restrictions we (as ChatGPT) cannot directly modify your local filesystem.  You will need to copy this PRD file into your `SCREEN TIME ANALYZER` repository manually.
- **Automation**: Provide a master script (e.g. `run_pipeline.py`) to orchestrate the entire flow given an episode.
- **Google Sheets Export**: Build a script to push totals and timelines to a Google Sheet via its API.
- **UI/Web App**: Develop a simple web or desktop app for uploading episodes, running the pipeline and viewing results.
- **MCP Tools**: Explore using other Model Context Protocol servers (e.g. shell MCP) for remote command execution, but note they currently require a secure, authenticated environment.

## 7 Initial Developer Steps
1. **Add this PRD**: Create a file named `PRD.md` (or `docs/PRD.md`) in the `SCREEN TIME ANALYZER` repository with the content above.  Commit it to version control.
2. **Set up the repository**:
   - Ensure the directory structure matches the layout described above.
   - Place model weights into `models/weights/` (e.g. YOLOv8n face, RetinaFace, ArcFace models).  Download from official sources if not already present.
   - Create configuration files `configs/pipeline.yaml` and `configs/bytetrack.yaml` with sensible defaults.
3. **Install dependencies**: Create a Python virtual environment and install packages such as `insightface`, `ultralytics` (for YOLO), `opencv-python`, `numpy`, `pandas`, `scipy`, `matplotlib`, `pydantic`, and any tracking libraries.
4. **Implement and test each stage**: Start with face detection and harvesting to validate detection quality.  Then add recognition, attribution and summarization modules.
5. **Build automation scripts**: Create CLI wrappers in `scripts/` (e.g. `scripts/harvest.py`, `scripts/build_facebank.py`, `scripts/associate.py`, `scripts/summarise.py`) to run the stages individually.

EOF