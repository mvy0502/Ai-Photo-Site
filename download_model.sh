#!/bin/bash
# Download MediaPipe FaceLandmarker model for V2 analyzer

echo "Downloading MediaPipe FaceLandmarker model..."
mkdir -p models

# Download blendshapes version (recommended)
echo "Downloading face_landmarker_v2_with_blendshapes.task..."
curl -L -o models/face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_blendshapes/float16/1/face_landmarker_v2_with_blendshapes.task

if [ $? -eq 0 ]; then
    echo "✅ Model downloaded successfully!"
    ls -lh models/face_landmarker_v2_with_blendshapes.task
else
    echo "❌ Download failed. Trying alternative model..."
    curl -L -o models/face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    if [ $? -eq 0 ]; then
        echo "✅ Alternative model downloaded!"
        ls -lh models/face_landmarker.task
    else
        echo "❌ All downloads failed"
        exit 1
    fi
fi

