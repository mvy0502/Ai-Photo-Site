#!/usr/bin/env python3
"""
Download MediaPipe FaceLandmarker model for V2 analyzer
"""

import os
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "blendshapes": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "basic": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    # Note: v2_with_blendshapes URL may not be available, using basic model instead
    # The basic model still works with V2 analyzer (blendshapes will use EAR fallback)
}

def download_model(model_type="blendshapes", show_progress=True):
    """Download MediaPipe model file"""
    if model_type not in MODEL_URLS:
        print(f"❌ Unknown model type: {model_type}")
        print(f"Available: {list(MODEL_URLS.keys())}")
        return False
    
    url = MODEL_URLS[model_type]
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Use basic model for both (v2_with_blendshapes URL not available)
    filename = "face_landmarker.task"
    if model_type == "blendshapes":
        # Try to use a more descriptive name, but download basic model
        print("⚠️  Note: Using basic face_landmarker model (v2_with_blendshapes not available)")
        print("   Blendshapes will use EAR fallback in V2 analyzer")
    
    output_path = models_dir / filename
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {output_path} ({size_mb:.1f} MB)")
        response = input("Download again? (y/N): ").strip().lower()
        if response != 'y':
            return True
    
    print(f"📥 Downloading {filename}...")
    print(f"   URL: {url}")
    
    try:
        def show_progress_hook(count, block_size, total_size):
            if show_progress and total_size > 0:
                percent = min(100, int(count * block_size * 100 / total_size))
                mb_downloaded = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=show_progress_hook if show_progress else None)
        
        if show_progress:
            print()  # New line after progress
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Download complete!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial file
        return False

if __name__ == "__main__":
    import sys
    
    model_type = "blendshapes"
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    print("=" * 60)
    print("MediaPipe FaceLandmarker Model Downloader")
    print("=" * 60)
    print()
    
    success = download_model(model_type)
    
    if success:
        print()
        print("=" * 60)
        print("✅ Model ready! You can now use V2 analyzer:")
        print("   USE_V2_ANALYZER=true uvicorn app:app --reload")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("❌ Download failed. Please check your internet connection.")
        print("=" * 60)
        sys.exit(1)

