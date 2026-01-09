#!/bin/bash
# V2 Analyzer Status Check Script

echo "=========================================="
echo "V2 ANALYZER STATUS CHECK"
echo "=========================================="
echo ""

# A) Check model file
echo "A) Model File Check:"
if [ -f "models/face_landmarker_v2_with_blendshapes.task" ]; then
    echo "  ✅ Model found: face_landmarker_v2_with_blendshapes.task"
    ls -lh models/face_landmarker_v2_with_blendshapes.task
elif [ -f "models/face_landmarker.task" ]; then
    echo "  ✅ Model found: face_landmarker.task"
    ls -lh models/face_landmarker.task
elif [ -f "models/face_landmarker_v2.task" ]; then
    echo "  ✅ Model found: face_landmarker_v2.task"
    ls -lh models/face_landmarker_v2.task
else
    echo "  ❌ No model file found!"
    echo "  Run: ./download_model.sh"
fi
echo ""

# B) Check if V2 is enabled
echo "B) Environment Check:"
if [ "$USE_V2_ANALYZER" = "true" ]; then
    echo "  ✅ USE_V2_ANALYZER=true"
else
    echo "  ⚠️  USE_V2_ANALYZER not set or false"
    echo "  Start app with: USE_V2_ANALYZER=true uvicorn app:app --reload"
fi
echo ""

# C) Test analyzer initialization
echo "C) Analyzer Initialization Test:"
python3 -c "
import os
os.environ['USE_V2_ANALYZER'] = 'true'
from utils.analyzer_v2 import get_analyzer
analyzer = get_analyzer()
if analyzer and analyzer.initialized:
    print('  ✅ Analyzer initialized successfully')
else:
    print('  ❌ Analyzer NOT initialized')
    print('  Check model file and MediaPipe installation')
" 2>&1 | sed 's/^/  /'
echo ""

echo "=========================================="
