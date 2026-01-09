#!/usr/bin/env python3
"""
Test script for Photo Test folder on Desktop
Analyzes selected test photos
"""
import sys
import os
from pathlib import Path
import json
import time

# Add current directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import analyze_image
from uuid import uuid4

# Test photo directory
TEST_PHOTOS_DIR = Path.home() / "Desktop" / "Photo Test"
OUTPUT_DIR = Path("test_results_photo_test")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_photo(photo_path):
    """Test a single photo and return results"""
    print(f"\n{'='*70}")
    print(f"📸 Testing: {photo_path.name}")
    print(f"{'='*70}")
    
    job_id = str(uuid4())
    
    try:
        # Run analysis
        result = analyze_image(job_id, str(photo_path))
        
        # Extract key information
        ok = result.get("ok", result.get("result") == "pass")
        issues = result.get("issues", [])
        metrics = result.get("metrics", {})
        
        # Print results
        status_icon = "✅" if ok else "❌"
        print(f"{status_icon} Status: {'PASS' if ok else 'FAIL'}")
        
        # Print key metrics
        print(f"\n📊 Key Metrics:")
        if "resolution_w" in metrics and "resolution_h" in metrics:
            print(f"   📐 Resolution: {metrics['resolution_w']}x{metrics['resolution_h']}")
        if "pixelation_score" in metrics:
            print(f"   🖼️  Pixelation Score: {metrics['pixelation_score']:.3f}")
        if "face_count" in metrics:
            print(f"   👤 Face Count: {metrics['face_count']}")
        if "blur_score" in metrics:
            print(f"   📷 Blur Score: {metrics['blur_score']:.2f}")
        if "brightness_mean" in metrics:
            print(f"   💡 Brightness: {metrics['brightness_mean']:.2f} ({metrics.get('brightness_status', 'unknown')})")
        if "face_area_ratio" in metrics:
            print(f"   📏 Face Area Ratio: {metrics['face_area_ratio']:.4f}")
        
        # Print all issues
        if issues:
            print(f"\n⚠️  Issues Found ({len(issues)}):")
            for issue in issues:
                severity_icon = "❌" if issue["severity"] == "fail" else "⚠️"
                print(f"   {severity_icon} [{issue['code']}] {issue['message']}")
        else:
            print(f"\n✅ No issues found - Photo is PASS!")
        
        # Save detailed results
        result_file = OUTPUT_DIR / f"{photo_path.stem}_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {result_file}")
        
        return {
            "photo": photo_path.name,
            "ok": ok,
            "issues_count": len(issues),
            "fail_count": len([i for i in issues if i["severity"] == "fail"]),
            "warn_count": len([i for i in issues if i["severity"] == "warn"]),
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        print(f"❌ Error analyzing photo: {e}")
        import traceback
        traceback.print_exc()
        return {
            "photo": photo_path.name,
            "ok": False,
            "error": str(e),
            "issues_count": 0
        }

def run_test_suite():
    """Run tests on all photos in Photo Test folder"""
    print("="*70)
    print("📸 Photo Test Folder - Detailed Analysis")
    print("="*70)
    
    if not TEST_PHOTOS_DIR.exists():
        print(f"❌ Folder not found: {TEST_PHOTOS_DIR}")
        return
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    photos = [f for f in TEST_PHOTOS_DIR.iterdir() 
              if f.suffix.lower() in image_extensions and f.is_file()]
    
    if not photos:
        print(f"❌ No photos found in {TEST_PHOTOS_DIR}")
        return
    
    print(f"\n📁 Folder: {TEST_PHOTOS_DIR}")
    print(f"📸 Found {len(photos)} photos to test")
    print(f"💾 Results will be saved to: {OUTPUT_DIR}\n")
    
    results = []
    
    # Test each photo
    for i, photo in enumerate(photos, 1):
        print(f"\n{'─'*70}")
        print(f"[{i}/{len(photos)}]")
        result = test_photo(photo)
        results.append(result)
        time.sleep(0.3)  # Small delay between tests
    
    # Summary report
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for r in results if r.get("ok", False))
    failed = total - passed
    
    print(f"\n📸 Total Photos: {total}")
    print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Issue breakdown
    all_issues = {}
    for result in results:
        for issue in result.get("issues", []):
            code = issue["code"]
            if code not in all_issues:
                all_issues[code] = {"count": 0, "severity": issue["severity"], "photos": []}
            all_issues[code]["count"] += 1
            all_issues[code]["photos"].append(result["photo"])
    
    if all_issues:
        print(f"\n⚠️  Issue Breakdown:")
        for code, info in sorted(all_issues.items(), key=lambda x: x[1]["count"], reverse=True):
            severity_icon = "❌" if info["severity"] == "fail" else "⚠️"
            print(f"   {severity_icon} {code}: {info['count']} photos")
            # Show first 3 photos with this issue
            if len(info["photos"]) <= 3:
                for photo_name in info["photos"]:
                    print(f"      - {photo_name}")
            else:
                for photo_name in info["photos"][:3]:
                    print(f"      - {photo_name}")
                print(f"      ... and {len(info['photos']) - 3} more")
    
    # Show passed photos
    passed_photos = [r["photo"] for r in results if r.get("ok", False)]
    if passed_photos:
        print(f"\n✅ Passed Photos ({len(passed_photos)}):")
        for photo_name in passed_photos:
            print(f"   - {photo_name}")
    
    # Save summary
    summary_file = OUTPUT_DIR / "test_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results,
            "issue_breakdown": all_issues,
            "passed_photos": passed_photos
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print("\n" + "="*70)
    print("✅ Test suite completed!")
    print("="*70)

if __name__ == "__main__":
    run_test_suite()

