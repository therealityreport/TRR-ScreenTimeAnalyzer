# Harvest Optimization - Summary of Changes

## What Was Done

### 1. Created Diagnostic Tools
- **`scripts/diagnose_harvest.py`** - Analyzes face detection and rejection reasons
- **`scripts/validate_harvest.py`** - Validates harvest output and generates statistics  
- **`scripts/compare_harvests.py`** - Compares before/after harvest results

### 2. Updated Configurations
- **`configs/pipeline.yaml`** - Optimized default config (better detection)
- **`configs/pipeline_permissive.yaml`** - Maximum coverage config
- **`configs/pipeline_original.yaml`** - Backup of original settings

### 3. Created Documentation
- **`docs/HARVEST_OPTIMIZATION.md`** - Complete guide with examples
- **`scripts/harvest_workflow.sh`** - Automated workflow script

## Key Configuration Changes

### Before (Original)
```yaml
stride: 2
face_conf_th: 0.40
min_area_frac: 0.005
face_in_track_iou: 0.10
dilate_track_px: 0.15
temporal_iou_tolerance: 1
```

### After (Optimized)
```yaml
stride: 1                    # Process every frame
face_conf_th: 0.30          # Lower threshold → more faces
min_area_frac: 0.002        # Smaller faces allowed
face_in_track_iou: 0.05     # Easier association
dilate_track_px: 0.25       # Larger person boxes
temporal_iou_tolerance: 3   # Look back ±3 frames
```

## Expected Improvements

Based on the changes, you should see:
- **2-3x more faces detected** (especially distant/profile faces)
- **Better track continuity** (fewer broken tracks)
- **Higher sample counts per track** (8-12 samples typically)
- **Fewer "UNKNOWN" labels** in final output

## New Runtime Controls (Oct 2025)

- `--threads N` caps CPU math/ONNX threads. The default is 1; `--fast` now forces 1 unless you override it explicitly.
- `--fast` bundles: stride=2, RetinaFace 640×640, `--defer-embeddings`, `--threads 1`, and disables debug rejection crops.
- CPU-only runs now auto-enable a preset when you leave `--stride` and `--retina-det-size` unset: stride bumps to at least 2 and RetinaFace drops to 640×640. Override either flag to opt out.
- `--min-frontalness` / `--frontalness-thresh` resolution now follows CLI flag → pipeline `min_frontalness` → 0.20 fallback, so configs can tune the default without requiring extra flags.
- `--defer-embeddings` skips ArcFace during harvest so you can generate embeddings later during clustering/facebank builds.
- `--no-identity-guard` bypasses the ArcFace purity guard and identity split checks—useful for quick passes when ArcFace is unavailable, but beware that mixed identities can slip into a single track.
- `--progress-interval` (percent) and `--heartbeat-sec` (seconds) tune log cadence. Default progress is 1% and heartbeat is 2 s.
- Every harvest now writes `progress.json` alongside the crops, e.g.
  ```json
  {
    "frames_total": 2479,
    "frames_done": 123,
    "percent": 4.96,
    "tracks": 3,
    "last_update": "2025-10-20T16:25:31Z"
  }
  ```
  and emits heartbeats so you get feedback within the first ~10 s even on CPU-only runs.
- Apple Silicon automatically prefers CoreML for RetinaFace and ArcFace; ONNX sessions fall back to CPU when CoreML is unavailable.

## Quick Start - Test the Changes

### Option 1: Automated Workflow (Easiest)
```bash
# Make script executable (first time only)
chmod +x scripts/harvest_workflow.sh

# Run full workflow
./scripts/harvest_workflow.sh data/RHOBH-TEST-v2.mp4
```

This will:
1. Run diagnostics
2. Run baseline harvest (original config)
3. Run optimized harvest (new config)
4. Compare results
5. Validate output

### Option 2: Manual Steps

#### Step 1: Diagnose
```bash
python scripts/diagnose_harvest.py \
    --video data/RHOBH-TEST-v2.mp4 \
    --sample 100 \
    --output diagnostics/test
```

#### Step 2: Run Harvest
```bash
python scripts/harvest_faces.py \
    data/RHOBH-TEST-v2.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --harvest-dir data/harvest/RHOBH-TEST-v2 \
    --threads 1 \
    --progress-interval 1 \
    --heartbeat-sec 2
```
> **Directory layout update:** When you pass `--harvest-dir`, the harvest now writes `track_*` folders directly inside the directory you provide. Use `--output-dir` (or the default) if you still want the legacy `<root>/<video_stem>/track_*` structure.
Add `--fast` when you need a quick low-heat pass (stride = 2, RetinaFace 640, deferred embeddings).

#### Step 3: Validate
```bash
python scripts/validate_harvest.py \
    data/harvest/RHOBH-TEST-v2 \
    --output diagnostics/validation
```

## What to Look For

### In Diagnostic Output:
```
DETECTION SUMMARY
==========================================
accepted                         345  (65.5%)
area_too_small                   120  (22.8%)
no_track_association              45  ( 8.5%)
low_confidence                    17  ( 3.2%)
```

**Good:** Acceptance rate > 60%  
**Needs tuning:** Rejection reasons > 30%

### In Validation Output:
```
STATISTICS:
Total tracks:          32
  Labeled:             6
  Unlabeled:           26
Total samples:         287
Actual image files:    287

QUALITY METRICS:
  Average quality:     0.687
  Average sharpness:   142.3
  Average frontalness: 0.521
```

**Good:** 
- Total samples > 200 for 90-second video
- Average quality > 0.6
- Labeled tracks match expected cast

## Troubleshooting

### "Still not finding enough faces"

Try the permissive config:
```bash
python scripts/harvest_faces.py \
    data/RHOBH-TEST-v2.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --pipeline-config configs/pipeline_permissive.yaml \
    --harvest-dir data/harvest/RHOBH-TEST-v2
```

### "Too many low-quality faces"

Use stricter thresholds or original config temporarily.

### "Faces detected but not associated with tracks"

Check diagnostics - look for high `no_track_association` count:
- Increase `dilate_track_px` to 0.30
- Increase `temporal_iou_tolerance` to 5
- Lower `face_in_track_iou` to 0.03

## Performance Notes

- **Processing time:** ~2-3x slower with stride=1 (but only needs to run once per episode)
- **Disk space:** More samples = more disk usage (typically 50-100MB per episode)
- **Quality:** The optimized config maintains quality while increasing coverage
- **Monitoring:** Follow `progress.json` or terminal heartbeats (default 2 s cadence) to track long CPU-bound harvests.

## Next Steps After Harvest

1. **Review samples**: 
   ```bash
   ls data/harvest/RHOBH-TEST-v2/track_*/
   ```

2. **Build facebank**:
   ```bash
   python scripts/build_facebank.py
   ```

3. **Run full tracking**:
   ```bash
   python scripts/run_tracker.py data/RHOBH-TEST-v2.mp4 \
       --person-weights models/weights/yolov8n.pt \
       --facebank-parquet data/facebank.parquet
   ```

## Files Created

### Scripts
- `scripts/diagnose_harvest.py` - Detection diagnostics
- `scripts/validate_harvest.py` - Harvest validation
- `scripts/compare_harvests.py` - Before/after comparison
- `scripts/harvest_workflow.sh` - Automated workflow

### Configs
- `configs/pipeline.yaml` - Optimized (new default)
- `configs/pipeline_permissive.yaml` - Maximum coverage
- `configs/pipeline_original.yaml` - Original backup

### Documentation
- `docs/HARVEST_OPTIMIZATION.md` - Complete guide
- `README_HARVEST_CHANGES.md` - This file

## Rollback Instructions

If you want to revert to original settings:

```bash
# Use original config explicitly
python scripts/harvest_faces.py \
    data/RHOBH-TEST-v2.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --pipeline-config configs/pipeline_original.yaml \
    --harvest-dir data/harvest/RHOBH-TEST-v2

# Or restore original as default
cp configs/pipeline_original.yaml configs/pipeline.yaml
```

---

**Questions?** See `docs/HARVEST_OPTIMIZATION.md` for detailed explanations.
