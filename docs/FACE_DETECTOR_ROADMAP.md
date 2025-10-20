# Face Detector Optimization Roadmap

## Current State
- **Detector:** InsightFace RetinaFace (buffalo_l)
- **Issue:** CoreML incompatible at 960×960 due to dynamic shapes
- **Workaround:** CPU-only or 640×640 CoreML

## Short-Term Fixes (< 30 min)

### Option A: 640×640 with CoreML
```bash
python scripts/harvest_faces.py VIDEO.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --retina-det-size 640 640 \
    --output-dir data/harvest
```
**Pros:** 2-3x faster than CPU, cooler  
**Cons:** May miss some distant faces

### Option B: CPU with Optimizations
```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python scripts/harvest_faces.py VIDEO.mp4 \
    --person-weights models/weights/yolov8n.pt \
    --retina-det-size 640 640 \
    --fast \
    --onnx-providers CPUExecutionProvider \
    --output-dir data/harvest
```
**Pros:** Coolest, most stable  
**Cons:** Slowest

## Long-Term Solutions (1-2 hours implementation)

### ⭐ Recommended: YOLOv8-face + CoreML

**Why YOLOv8-face:**
- Already using YOLO for person detection
- Export to CoreML natively (no ONNX compatibility issues)
- Fast, accurate, runs cool on Apple Neural Engine
- Single model architecture for both person + face

**Implementation steps:**
1. Download YOLOv8-face weights
2. Export to CoreML format
3. Create `YOLOFaceDetector` class (similar to `YOLOPersonDetector`)
4. Swap in `harvest.py`

**Code structure:**
```python
# screentime/detectors/face_yolo.py
from ultralytics import YOLO

class YOLOFaceDetector:
    def __init__(self, weights: str, conf_thres: float = 0.3):
        self.model = YOLO(weights)  # .mlpackage for CoreML
    
    def detect(self, image, frame_idx):
        results = self.model(image, conf=self.conf_thres)
        # Convert to Detection format
        # Extract landmarks if model provides them
```

**Estimated time:** 1-2 hours  
**Benefit:** 3-5x faster than current CPU setup, runs cool

---

### Alternative: Apple Vision Framework

**Why Vision:**
- Native CoreML, zero compatibility issues
- Built into macOS, no extra dependencies
- Rock-solid stability

**Cons:**
- macOS-only (not portable to Linux servers)
- Less accurate than YOLO/RetinaFace
- No facial landmarks (would need separate alignment step)

**Implementation:**
```python
# screentime/detectors/face_vision.py
import Vision
import CoreImage

class VisionFaceDetector:
    def detect(self, image, frame_idx):
        # Use VNDetectFaceRectanglesRequest
        # Returns bounding boxes only
```

**Estimated time:** 2-3 hours  
**Benefit:** Extremely stable, cool running

---

### Alternative: MediaPipe Face Detection

**Why MediaPipe:**
- Google's solution, optimized for real-time
- GPU/Metal acceleration on Mac
- Cross-platform (Mac, Linux, mobile)

**Cons:**
- Another dependency
- Conversion layer needed for existing pipeline

**Estimated time:** 2-3 hours  
**Benefit:** Good balance of speed + portability

---

## Recommendation Priority

1. **Today:** Use 640×640 with CoreML for testing
2. **This week:** Complete harvest/facebank workflow with current setup
3. **Next optimization:** Implement YOLOv8-face + CoreML export
4. **Measure:** Compare speed/accuracy/heat before/after

---

## Resources

- YOLOv8-face models: https://github.com/akanametov/yolov8-face
- Ultralytics CoreML export: https://docs.ultralytics.com/modes/export/#arguments
- Apple Vision docs: https://developer.apple.com/documentation/vision/vndetectfacerectanglesrequest
