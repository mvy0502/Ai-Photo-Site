#!/usr/bin/env python3
"""Test script to verify sunglasses detection with real photo"""

import os
import sys
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set V2 analyzer
os.environ['USE_V2_ANALYZER'] = 'true'

from utils.analyze_v2 import analyze_image_v2

def test_sunglasses_photo():
    """Test the sunglasses photo"""
    # Try multiple possible locations
    test_paths = [
        os.path.expanduser('~/Desktop/Photo Test/inst_dark_tinted_lenses.webp'),
        'uploads/570f1677-c57a-4a5f-b9be-68e4ccfdcad1.webp',  # Latest upload
    ]
    
    image_path = None
    for path in test_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print("❌ Could not find test image")
        print("   Tried:")
        for p in test_paths:
            print(f"     - {p}")
        return False
    
    print(f"✅ Found test image: {image_path}")
    print(f"   Size: {os.path.getsize(image_path)} bytes")
    
    # Read image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Could not read image: {image_path}")
        return False
    
    print(f"✅ Image loaded: {image_bgr.shape}")
    
    # Analyze
    print("\n" + "="*60)
    print("Running analysis...")
    print("="*60)
    
    try:
        # analyze_image_v2 expects (job_id, file_path) but we have image array
        # Let's check the function signature
        import inspect
        sig = inspect.signature(analyze_image_v2)
        print(f"Function signature: {sig}")
        
        # Try calling with file path
        result = analyze_image_v2("test_job", image_path)
        
        # Check results
        issues = result.get("issues", [])
        metrics = result.get("metrics", {})
        
        print(f"\n📊 Metrics:")
        print(f"  sunglasses_score: {metrics.get('sunglasses_score', 'N/A')}")
        print(f"  iris_visibility: {metrics.get('iris_visibility', 'N/A')}")
        print(f"  hair_occlusion_score: {metrics.get('hair_occlusion_score', 'N/A')}")
        
        print(f"\n🚨 Issues ({len(issues)}):")
        for issue in issues:
            code = issue.get("code", "UNKNOWN")
            message = issue.get("message", "")
            severity = issue.get("severity", "unknown")
            print(f"  [{severity.upper()}] {code}: {message}")
        
        # Check if sunglasses detected
        has_sunglasses_issue = any(i.get("code") == "SUNGLASSES" for i in issues)
        has_hair_issue = any(i.get("code") == "HAIR_OVER_EYES" for i in issues)
        
        print(f"\n✅ Analysis:")
        print(f"  Sunglasses detected: {has_sunglasses_issue}")
        print(f"  Hair over eyes detected: {has_hair_issue}")
        
        # Expected: Sunglasses should be detected, hair should NOT be detected
        if has_sunglasses_issue and not has_hair_issue:
            print(f"\n✅✅✅ CORRECT: Sunglasses detected, hair check skipped!")
            return True
        elif has_sunglasses_issue and has_hair_issue:
            print(f"\n❌❌❌ ERROR: Both sunglasses AND hair detected (should only be sunglasses)")
            print(f"   This is the bug we need to fix!")
            return False
        elif not has_sunglasses_issue and has_hair_issue:
            print(f"\n❌❌❌ ERROR: Hair detected but sunglasses NOT detected")
            print(f"   sunglasses_score: {metrics.get('sunglasses_score', 'N/A')}")
            print(f"   iris_visibility: {metrics.get('iris_visibility', 'N/A')}")
            return False
        else:
            print(f"\n⚠️  WARNING: Unexpected state")
            print(f"   sunglasses_score: {metrics.get('sunglasses_score', 'N/A')}")
            print(f"   iris_visibility: {metrics.get('iris_visibility', 'N/A')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("🔍 Testing Sunglasses Detection")
    print("="*60)
    print()
    
    success = test_sunglasses_photo()
    
    print("\n" + "="*60)
    if success:
        print("✅✅✅ TEST PASSED!")
    else:
        print("❌❌❌ TEST FAILED - NEEDS FIXING")
    print("="*60)

