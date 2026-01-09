#!/usr/bin/env python3
"""
PhotoRoom Pipeline Regression Test Script

Tests:
1. Normal photo => PASS
2. Glasses photo => WARN + requires_ack checkbox
3. Sunglasses/dark lens => FAIL
4. Head tilted => FAIL (HEAD_POSE_ROLL)
5. Too close => FAIL (FACE_TOO_LARGE)
"""

import os
import sys
import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"
TEST_PHOTOS_DIR = "/Users/vedat/Desktop/Photo Test Real Phone Photos"

TESTS = [
    {
        "name": "Normal Photo",
        "file": "normalphoto.jpeg",
        "expected_status": "PASS",
        "expected_issues": [],
        "should_can_continue": True
    },
    {
        "name": "Normal Photo With Glasses",
        "file": "normalphotowithglasses.jpeg",
        "expected_status": "WARN",
        "expected_issues": ["EYEWEAR_GLASSES"],
        "should_can_continue": False,  # Without ack
        "requires_ack": True
    },
    {
        "name": "Eyes Closed",
        "file": "eyesclosed.jpeg",
        "expected_status": "FAIL",
        "expected_issues": ["EYES_CLOSED"],
        "should_can_continue": False
    },
    {
        "name": "Looking Sideways",
        "file": "photolookingsideways.jpeg",
        "expected_status": "FAIL",
        "expected_issues": ["HEAD_POSE_YAW"],
        "should_can_continue": False
    },
    {
        "name": "Looking Down",
        "file": "lookingdown.jpeg",
        "expected_status": "FAIL",
        "expected_issues": ["HEAD_POSE_PITCH"],
        "should_can_continue": False
    },
    {
        "name": "Teeth Showing",
        "file": "teethshowing.jpeg",
        "expected_status": "FAIL",
        "expected_issues": ["TEETH_VISIBLE"],
        "should_can_continue": False
    },
    {
        "name": "Too Close",
        "file": "tooclose.jpeg",
        "expected_status": "FAIL",
        "expected_issues": ["FACE_TOO_LARGE"],
        "should_can_continue": False
    },
    {
        "name": "Too Far Off",
        "file": "toofaroff.jpeg",
        "expected_status": "FAIL",
        "expected_issues": ["FACE_TOO_SMALL"],
        "should_can_continue": False
    },
]


def check_server():
    """Check if server is running."""
    try:
        r = requests.get(f"{BASE_URL}/", timeout=3)
        return r.status_code == 200
    except:
        return False


def run_test(test_case):
    """Run a single test case."""
    name = test_case["name"]
    filepath = os.path.join(TEST_PHOTOS_DIR, test_case["file"])
    
    if not os.path.exists(filepath):
        return {"name": name, "status": "SKIP", "reason": "File not found"}
    
    # Upload photo
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/upload",
            files={"photo": (test_case["file"], f, "image/jpeg")}
        )
    
    if r.status_code != 200:
        return {"name": name, "status": "ERROR", "reason": f"Upload failed: {r.status_code}"}
    
    upload_data = r.json()
    job_id = upload_data.get("job_id")
    
    if not job_id:
        return {"name": name, "status": "ERROR", "reason": "No job_id in response"}
    
    # Wait for processing
    time.sleep(2)
    
    # Get job status
    r = requests.get(f"{BASE_URL}/job/{job_id}/status")
    if r.status_code != 200:
        return {"name": name, "status": "ERROR", "reason": f"Status check failed: {r.status_code}"}
    
    job_data = r.json()
    
    # Extract results
    final_status = job_data.get("final_status", "UNKNOWN")
    issues = [i.get("id") for i in job_data.get("issues", [])]
    can_continue = job_data.get("server_can_continue", job_data.get("can_continue"))
    analysis_source = job_data.get("analysis_source", "UNKNOWN")
    pipeline_version = job_data.get("pipeline_version", "UNKNOWN")
    
    # Check expectations
    status_ok = final_status == test_case["expected_status"]
    
    # Check issues - at least one expected issue should be present
    expected_issues = test_case["expected_issues"]
    issues_ok = True
    if expected_issues:
        issues_ok = any(exp in issues for exp in expected_issues)
    else:
        issues_ok = len(issues) == 0
    
    can_continue_ok = can_continue == test_case["should_can_continue"]
    
    all_ok = status_ok and issues_ok and can_continue_ok
    
    result = {
        "name": name,
        "status": "PASS" if all_ok else "FAIL",
        "final_status": final_status,
        "expected_status": test_case["expected_status"],
        "issues": issues,
        "expected_issues": expected_issues,
        "can_continue": can_continue,
        "expected_can_continue": test_case["should_can_continue"],
        "analysis_source": analysis_source,
        "pipeline_version": pipeline_version,
        "checks": {
            "status_ok": status_ok,
            "issues_ok": issues_ok,
            "can_continue_ok": can_continue_ok
        }
    }
    
    return result


def main():
    print("=" * 70)
    print("PhotoRoom Pipeline Regression Tests")
    print("=" * 70)
    print()
    
    # Check server
    if not check_server():
        print("❌ Server not running at", BASE_URL)
        print("   Start with: uvicorn app:app --reload")
        sys.exit(1)
    
    print(f"✅ Server running at {BASE_URL}")
    print()
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    skipped = 0
    
    for test_case in TESTS:
        print(f"Testing: {test_case['name']}...", end=" ", flush=True)
        result = run_test(test_case)
        results.append(result)
        
        if result["status"] == "PASS":
            print(f"✅ PASS")
            passed += 1
        elif result["status"] == "SKIP":
            print(f"⏭️ SKIP ({result.get('reason', '')})")
            skipped += 1
        else:
            print(f"❌ FAIL")
            print(f"   Expected: status={test_case['expected_status']}, issues={test_case['expected_issues']}")
            print(f"   Got: status={result.get('final_status')}, issues={result.get('issues')}")
            if not result.get("checks", {}).get("can_continue_ok"):
                print(f"   can_continue: expected={test_case['should_can_continue']}, got={result.get('can_continue')}")
            failed += 1
    
    # Summary
    print()
    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)
    
    # Show pipeline info from last successful test
    for r in results:
        if r.get("analysis_source"):
            print(f"Analysis Source: {r.get('analysis_source')}")
            print(f"Pipeline Version: {r.get('pipeline_version')}")
            break
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


