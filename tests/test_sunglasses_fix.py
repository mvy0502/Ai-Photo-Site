#!/usr/bin/env python3
"""Test script to verify sunglasses detection and hair check skip"""

import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.analyze_v2 import analyze_image_v2

def find_test_images():
    """Find test images, especially sunglasses ones"""
    test_dirs = ['tests', '.', 'test_images']
    image_exts = ['.jpg', '.jpeg', '.png', '.webp']
    keywords = ['sunglass', 'tinted', 'glass', 'dark_tinted', 'inst_dark']
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue
        for file in os.listdir(test_dir):
            if any(file.lower().endswith(ext) for ext in image_exts):
                file_lower = file.lower()
                if any(keyword in file_lower for keyword in keywords):
                    full_path = os.path.join(test_dir, file)
                    yield full_path

def test_image(image_path):
    """Test a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return False
    
    # Read image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Could not read image: {image_path}")
        return False
    
    print(f"✅ Image loaded: {image_bgr.shape}")
    
    # Analyze
    try:
        result = analyze_image_v2(image_bgr, {})
        
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
            return False
        elif not has_sunglasses_issue and has_hair_issue:
            print(f"\n❌❌❌ ERROR: Hair detected but sunglasses NOT detected")
            return False
        else:
            print(f"\n⚠️  WARNING: Unexpected state")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔍 Searching for sunglasses test images...")
    
    test_images = list(find_test_images())
    
    if not test_images:
        print("❌ No sunglasses test images found!")
        print("   Looking in: tests/, ., test_images/")
        print("   Keywords: sunglass, tinted, glass, dark_tinted, inst_dark")
        return
    
    print(f"✅ Found {len(test_images)} test image(s)")
    
    all_passed = True
    for image_path in test_images:
        if not test_image(image_path):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅✅✅ ALL TESTS PASSED!")
    else:
        print("❌❌❌ SOME TESTS FAILED - NEEDS FIXING")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

