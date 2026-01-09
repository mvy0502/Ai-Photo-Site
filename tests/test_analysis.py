#!/usr/bin/env python3
"""
Comprehensive test script for photo analysis
Tests all new Sprint 2 checks on local photos
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
TEST_PHOTOS_DIR = Path("uploads")
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_photo(photo_path):
    """Test a single photo and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {photo_path.name}")
    print(f"{'='*60}")
    
    job_id = str(uuid4())
    
    try:
        # Run analysis
        result = analyze_image(job_id, str(photo_path))
        
        # Extract key information
        ok = result.get("ok", result.get("result") == "pass")
        issues = result.get("issues", [])
        metrics = result.get("metrics", {})
        
        # Print results
        print(f"✅ Status: {'PASS' if ok else 'FAIL'}")
        print(f"📊 Metrics:")
        for key, value in metrics.items():
            print(f"   - {key}: {value}")
        
        if issues:
            print(f"⚠️  Issues ({len(issues)}):")
            for issue in issues:
                severity_icon = "❌" if issue["severity"] == "fail" else "⚠️"
                print(f"   {severity_icon} [{issue['code']}] {issue['message']}")
        else:
            print("✅ No issues found")
        
        # Save detailed results
        result_file = OUTPUT_DIR / f"{photo_path.stem}_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed results saved to: {result_file}")
        
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
    """Run tests on all photos in uploads directory"""
    print("="*60)
    print("Sprint 2 - Photo Analysis Test Suite")
    print("="*60)
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    photos = [f for f in TEST_PHOTOS_DIR.iterdir() 
              if f.suffix.lower() in image_extensions and f.is_file()]
    
    if not photos:
        print(f"❌ No photos found in {TEST_PHOTOS_DIR}")
        return
    
    print(f"\n📸 Found {len(photos)} photos to test")
    print(f"📁 Testing photos from: {TEST_PHOTOS_DIR}")
    print(f"💾 Results will be saved to: {OUTPUT_DIR}\n")
    
    results = []
    
    # Test each photo
    for i, photo in enumerate(photos, 1):
        print(f"\n[{i}/{len(photos)}] Processing...")
        result = test_photo(photo)
        results.append(result)
        time.sleep(0.5)  # Small delay between tests
    
    # Summary report
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results if r.get("ok", False))
    failed = total - passed
    
    print(f"\n📊 Total Photos: {total}")
    print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Issue breakdown
    all_issues = {}
    for result in results:
        for issue in result.get("issues", []):
            code = issue["code"]
            if code not in all_issues:
                all_issues[code] = {"count": 0, "severity": issue["severity"]}
            all_issues[code]["count"] += 1
    
    if all_issues:
        print(f"\n⚠️  Issue Breakdown:")
        for code, info in sorted(all_issues.items(), key=lambda x: x[1]["count"], reverse=True):
            severity_icon = "❌" if info["severity"] == "fail" else "⚠️"
            print(f"   {severity_icon} {code}: {info['count']} photos")
    
    # Save summary
    summary_file = OUTPUT_DIR / "test_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results,
            "issue_breakdown": all_issues
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print("\n" + "="*60)
    print("Test suite completed!")
    print("="*60)

if __name__ == "__main__":
    run_test_suite()

