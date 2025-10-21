# Harvest Optimization Guide

## Overview

This guide explains how to use the new diagnostic and validation tools to improve face detection and harvest quality.

## Quick Start

### 1. Diagnose Detection Issues

First, run diagnostics to see what's being detected and what's being rejected:

```bash
# Analyze a sample of frames
python scripts/diagnose_harvest.py \
    --video data/RHOBH-TEST.mp4 \
    --sample 100 \
    --output diagnostics/rhobh_test

# Or analyze a specific frame range
python scripts/diagnose_harvest.py \
    --video data/RHOBH-TEST.mp4 \
    --frames 500-700 \
    --save-frames \
    --output diagnostics/rhobh_test
```

This will:
- Detect faces and persons in sampled frames
- Show rejection reasons (too small, low confidence, no track association)
- Generate recommendations for config adjustments
- Optionally save annotated frames for visual inspection

**Output:**
- `diagnostics/rhobh_test/detection_log.csv` - Detailed detection data
- `diagnostics/rhobh_test/rejection_summary.csv` - Rejection reason counts
- `diagnostics/rhobh_test/frame_*.jpg` - Annotated frames (if --save-frames used)

### 2. Run Harvest with Original Config (Baseline)

```bash
# Use original config
python scripts/harvest_faces.py \
    data/RHOBH-TEST.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --pipeline-config configs/pipeline_original.yaml \
    --output-dir data/harvest

# Move results for comparison
mv data/harvest/RHOBH-TEST data/harvest/RHOBH-TEST_original
```

### 3. Run Harvest with Optimized Config

```bash
# Use optimized config (now the default)
python scripts/harvest_faces.py \
    data/RHOBH-TEST.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --output-dir data/harvest
```

### 4. Compare Results

```bash
python scripts/compare_harvests.py \
    --before data/harvest/RHOBH-TEST_original \
    --after data/harvest/RHOBH-TEST \
    --output diagnostics/harvest_comparison
```

This generates:
- Console summary showing improvement metrics
- `diagnostics/harvest_comparison/comparison.csv` - Detailed comparison
- `diagnostics/harvest_comparison/report.txt` - Summary report

### 5. Validate Harvest

```bash
# Quick validation
python scripts/validate_harvest.py data/harvest/RHOBH-TEST

# Detailed validation with per-track stats
python scripts/validate_harvest.py \
    data/harvest/RHOBH-TEST \
    --show-tracks \
    --output diagnostics/validation
```

## Configuration Files

### `configs/pipeline.yaml` (Default - Optimized)

**Recommended for:** Most episodes after initial testing

Key improvements:
- `stride: 1` - Process every frame
- `face_conf_th: 0.30` - Lower threshold (was 0.40)
- `min_area_frac: 0.002` - Allow smaller faces (was 0.005)
- `face_in_track_iou: 0.05` - More permissive association (was 0.10)
- `dilate_track_px: 0.25` - Larger person box expansion (was 0.15)
- `temporal_iou_tolerance: 3` - Look back ±3 frames (was 1)

### `configs/pipeline_permissive.yaml`

**Recommended for:** First-time harvest, building initial facebank, difficult videos

Even more aggressive detection:
- `face_conf_th: 0.25` - Very low threshold
- `min_area_frac: 0.001` - Accept very small faces
- `face_in_track_iou: 0.03` - Minimal overlap required
- `temporal_iou_tolerance: 5` - Look back ±5 frames
- `debug_rejections: true` - Enable detailed logging

Usage:
```bash
python scripts/harvest_faces.py \
    data/video.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --pipeline-config configs/pipeline_permissive.yaml \
    --output-dir data/harvest
```

### `configs/pipeline_original.yaml`

**Backup of original settings** - Use for comparison or if new settings cause issues

## Understanding Rejection Reasons

When running diagnostics, you'll see these rejection reasons:

| Reason | Meaning | Fix |
|--------|---------|-----|
| `area_too_small` | Face bounding box is too small | Lower `min_area_frac` |
| `low_confidence` | Face detector confidence is low | Lower `face_conf_th` |
| `no_track_association` | Face doesn't overlap with any person track | Lower `face_in_track_iou`, increase `dilate_track_px`, or increase `temporal_iou_tolerance` |
| `rejected_frontalness` | Face is too profile/side-view | Lower `min_frontalness` |
| `rejected_sharpness` | Face is too blurry | Lower `min_sharpness_laplacian` or `sharpness_pctile` |

> **Note:** Sharpness is computed from BGR frames by converting them to grayscale with OpenCV. If you preprocess frames yourself, be sure to keep them in BGR order (OpenCV's default) or convert back before running harvest diagnostics to avoid under-estimating sharpness.

## Interpreting Results

### Good Harvest Metrics

✓ **Total samples:** 200-500+ faces for a 90-second video
✓ **Samples per track:** Average 8-12
✓ **Quality scores:** Average > 0.6
✓ **Acceptance rate:** > 60% in diagnostics

### Warning Signs

⚠️ **Too few samples:** < 100 faces for a 90-second video with multiple people
⚠️ **Empty tracks:** > 10% of tracks have no samples
⚠️ **High rejection rate:** > 50% of detected faces rejected
⚠️ **Quality drop:** Average quality < 0.5 (too permissive)

## Workflow for New Videos

1. **Quick diagnostic** (5 minutes):
   ```bash
   python scripts/diagnose_harvest.py --video NEW_VIDEO.mp4 --sample 50
   ```

2. **Adjust config** if needed based on recommendations

3. **Run harvest**:
   ```bash
   python scripts/harvest_faces.py NEW_VIDEO.mp4 \
       --person-weights models/weights/yolov8n.pt
   ```

4. **Validate**:
   ```bash
   python scripts/validate_harvest.py data/harvest/NEW_VIDEO
   ```

5. **Proceed to facebank** if validation looks good

## Troubleshooting

### Issue: "Very few faces detected"

**Solution:**
1. Run diagnostics with `--save-frames` to see what's happening
2. Check if faces are being detected but rejected
3. Use `configs/pipeline_permissive.yaml` for more aggressive detection

### Issue: "Too many low-quality faces"

**Solution:**
1. Check average quality scores in validation
2. If < 0.5, increase quality thresholds
3. Consider using stricter config after initial discovery

### Issue: "Faces not associating with tracks"

**Solution:**
1. Check `track_iou` values in detection log
2. Increase `dilate_track_px` (try 0.30)
3. Increase `temporal_iou_tolerance` (try 5)
4. Lower `face_in_track_iou` (try 0.03)

### Issue: "Same person split across multiple tracks"

**Solution:**
1. Check ByteTrack config (configs/bytetrack.yaml)
2. Increase `track_buffer` (try 35)
3. Lower `match_thresh` (try 0.80)

## Performance Notes

- **Stride = 1** is slower but comprehensive (recommended for harvest)
- **Stride = 2** is 2x faster, may miss brief appearances
- Detection at 960x960 is accurate but slow (30-40 fps on CPU)
- Consider 640x640 for faster initial testing (but may miss distant faces)

## Next Steps

After successful harvest:
1. Review harvested faces: `ls data/harvest/VIDEO/track_*/`
2. Build/update facebank: `python scripts/build_facebank.py`
3. Run tracker: `python scripts/run_tracker.py`

---

**Last Updated:** 2025-10-16
**Config Version:** Optimized v1.0
