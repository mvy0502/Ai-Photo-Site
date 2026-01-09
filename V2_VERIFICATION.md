# V2 Analyzer Verification Guide

## 3 Critical Checks

### A) Model File Check
```bash
ls -lh models/
```

**Expected output:**
```
face_landmarker_v2_with_blendshapes.task  (several MB, e.g., 9.2M)
```

**If missing:**
```bash
./download_model.sh
```

### B) Start App with V2
```bash
USE_V2_ANALYZER=true uvicorn app:app --reload
```

### C) Check Startup Logs

**✅ V2 Active (what you should see):**
```
============================================================
🔵 V2 PROFESSIONAL ANALYZER ENABLED
   Using MediaPipe Tasks API with FaceLandmarker
============================================================
✅ [V2] FaceLandmarker initialized with blendshapes support
   Model: models/face_landmarker_v2_with_blendshapes.task
```

**❌ V2 NOT Active (fallback to V1):**
```
============================================================
🔵 V2 PROFESSIONAL ANALYZER ENABLED
   Using MediaPipe Tasks API with FaceLandmarker
============================================================
⚠️  Warning: FaceLandmarker model file not found...
```

## Quick Status Check

```bash
./check_v2_status.sh
```

## Verify V2 Metrics in Results

When analyzing a photo, check the JSON response for V2-specific metrics:

```json
{
  "metrics": {
    "head_pose_yaw": -2.5,
    "head_pose_pitch": 1.2,
    "head_pose_roll": 0.8,
    "eyes_open_prob": 0.95,
    "gaze_forward_prob": 0.88,
    "iris_visibility": 0.92,
    "sunglasses_score": 0.1,
    "hat_score": 0.05,
    "hair_occlusion_score": 0.02
  }
}
```

**If these metrics are missing:** V2 is not working, check model file and logs.
