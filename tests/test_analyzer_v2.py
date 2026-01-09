"""
Test suite for V2 analyzer - golden set tests
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.analyze_v2 import analyze_image_v2

# Test photos directory
TEST_PHOTOS_DIR = Path.home() / "Desktop/Photo Test"

# Expected results for golden set
EXPECTED_RESULTS = {
    "goodphoto.jpg": {
        "ok": True,
        "result": "pass",
        "should_have_issues": []
    },
    "inst_blurred.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["BLURRED"]
    },
    "inst_too_dark.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["TOO_DARK"]
    },
    "inst_looking_away.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["NOT_LOOKING_FORWARD"]
    },
    "inst_hair_across_eyes.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["HAIR_OVER_EYES"]
    },
    "inst_closed_eyes.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["EYES_CLOSED"]
    },
    "inst_dark_tinted_lenses.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["SUNGLASSES"]
    },
    # Add a glasses test case (if you have a test image with regular glasses)
    # "inst_glasses.webp": {
    #     "ok": False,
    #     "result": "fail",
    #     "should_have_issues": ["GLASSES"]
    # },
    "inst_wearing_hat.webp": {
        "ok": False,
        "result": "fail",
        "should_have_issues": ["HAT"]
    }
}


@pytest.mark.parametrize("photo_name,expected", EXPECTED_RESULTS.items())
def test_golden_set(photo_name, expected):
    """Test each photo in golden set against expected results"""
    photo_path = TEST_PHOTOS_DIR / photo_name
    
    if not photo_path.exists():
        pytest.skip(f"Test photo not found: {photo_path}")
    
    # Run analysis
    result = analyze_image_v2("test", str(photo_path))
    
    # Check basic structure
    assert result is not None, f"Analysis returned None for {photo_name}"
    assert "ok" in result, f"Result missing 'ok' field for {photo_name}"
    assert "result" in result, f"Result missing 'result' field for {photo_name}"
    assert "issues" in result, f"Result missing 'issues' field for {photo_name}"
    
    # Check expected result
    assert result["ok"] == expected["ok"], \
        f"{photo_name}: Expected ok={expected['ok']}, got {result['ok']}"
    assert result["result"] == expected["result"], \
        f"{photo_name}: Expected result={expected['result']}, got {result['result']}"
    
    # Check expected issues
    issue_codes = [issue["code"] for issue in result.get("issues", [])]
    
    for expected_issue in expected["should_have_issues"]:
        assert expected_issue in issue_codes, \
            f"{photo_name}: Expected issue '{expected_issue}' not found. Found: {issue_codes}"
    
    # For PASS photos, should have no FAIL issues (WARN is OK)
    if expected["ok"]:
        fail_issues = [i for i in result.get("issues", []) if i.get("severity") == "fail"]
        # Special check: goodphoto.jpg must NOT have SUNGLASSES FAIL
        if photo_name == "goodphoto.jpg":
            sunglasses_fail = [i for i in fail_issues if i.get("code") == "SUNGLASSES"]
            assert len(sunglasses_fail) == 0, \
                f"{photo_name}: Normal photo must NOT have SUNGLASSES FAIL. Found: {sunglasses_fail}"
        assert len(fail_issues) == 0, \
            f"{photo_name}: Expected PASS but found FAIL issues: {[i['code'] for i in fail_issues]}"
    
    # Special check: inst_dark_tinted_lenses.webp must have SUNGLASSES FAIL (not WARN)
    if photo_name == "inst_dark_tinted_lenses.webp":
        sunglasses_issues = [i for i in result.get("issues", []) if i.get("code") == "SUNGLASSES"]
        assert len(sunglasses_issues) > 0, \
            f"{photo_name}: Must have SUNGLASSES issue. Found issues: {[i['code'] for i in result.get('issues', [])]}"
        # Check that it's FAIL, not WARN
        sunglasses_fail = [i for i in sunglasses_issues if i.get("severity") == "fail"]
        assert len(sunglasses_fail) > 0, \
            f"{photo_name}: SUNGLASSES must be FAIL, not WARN. Found: {[i['severity'] for i in sunglasses_issues]}"


def test_analyzer_availability():
    """Test that analyzer can be initialized"""
    from utils.analyzer_v2 import get_analyzer
    analyzer = get_analyzer()
    # Analyzer may be None if model file is missing, which is OK for tests
    # Just check that function doesn't crash
    assert True


# =============================================================================
# Glasses/Sunglasses Policy Tests
# =============================================================================

class TestGlassesSunglassesPolicy:
    """
    Test the glasses/sunglasses policy:
    - SUNGLASSES (dark/tinted lenses) => FAIL (blocking)
    - GLASSES (clear/normal glasses) => WARN (non-blocking, requires acknowledgement)
    - If both detected, SUNGLASSES wins
    - Frame presence alone should NOT trigger SUNGLASSES
    """
    
    def test_sunglasses_photo_policy(self):
        """Test: Sunglasses photo => SUNGLASSES fail, policy.can_proceed=false"""
        photo_path = TEST_PHOTOS_DIR / "inst_dark_tinted_lenses.webp"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_sunglasses_policy", str(photo_path))
        
        # Check that SUNGLASSES issue is present with severity=fail
        issues = result.get("issues", [])
        sunglasses_issues = [i for i in issues if i["code"] == "SUNGLASSES"]
        
        assert len(sunglasses_issues) > 0, "SUNGLASSES issue should be present"
        assert sunglasses_issues[0]["severity"] == "fail", "SUNGLASSES should have severity=fail"
        
        # Check policy
        policy = result.get("policy", {})
        assert policy.get("can_proceed") == False, "can_proceed should be False for sunglasses"
        assert policy.get("requires_ack") == False, "requires_ack should be False (sunglasses blocks, no ack needed)"
        assert policy.get("ack_code") is None, "ack_code should be None"
        
        # Check that GLASSES is NOT present (sunglasses wins)
        glasses_issues = [i for i in issues if i["code"] == "GLASSES"]
        assert len(glasses_issues) == 0, "GLASSES should NOT be present when SUNGLASSES is detected"
    
    def test_sunglasses_message_correct(self):
        """Test: Sunglasses message is in Turkish and includes instruction to remove"""
        photo_path = TEST_PHOTOS_DIR / "inst_dark_tinted_lenses.webp"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_sunglasses_message", str(photo_path))
        
        issues = result.get("issues", [])
        sunglasses_issues = [i for i in issues if i["code"] == "SUNGLASSES"]
        
        if len(sunglasses_issues) > 0:
            message = sunglasses_issues[0]["message"]
            assert "çıkarın" in message.lower() or "çıkar" in message.lower(), \
                f"Sunglasses message should include instruction to remove. Got: {message}"
    
    def test_good_photo_policy(self):
        """Test: Good photo => no GLASSES/SUNGLASSES, can_proceed=true, requires_ack=false"""
        photo_path = TEST_PHOTOS_DIR / "goodphoto.jpg"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_good_policy", str(photo_path))
        
        # Check no glasses/sunglasses issues
        issues = result.get("issues", [])
        glasses_codes = ["GLASSES", "SUNGLASSES"]
        glasses_issues = [i for i in issues if i["code"] in glasses_codes]
        
        assert len(glasses_issues) == 0, f"Good photo should have no glasses issues. Found: {glasses_issues}"
        
        # Check policy allows proceeding
        policy = result.get("policy", {})
        assert policy.get("can_proceed") == True, "can_proceed should be True for good photo"
        assert policy.get("requires_ack") == False, "requires_ack should be False for good photo"
        assert policy.get("ack_code") is None, "ack_code should be None for good photo"
    
    def test_policy_structure(self):
        """Test: Policy object has correct structure"""
        photo_path = TEST_PHOTOS_DIR / "goodphoto.jpg"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_policy_structure", str(photo_path))
        
        # Check policy exists and has required fields
        assert "policy" in result, "Result should contain 'policy' field"
        
        policy = result["policy"]
        assert "can_proceed" in policy, "Policy should have 'can_proceed' field"
        assert "requires_ack" in policy, "Policy should have 'requires_ack' field"
        assert "ack_code" in policy, "Policy should have 'ack_code' field"
        
        # Check types
        assert isinstance(policy["can_proceed"], bool), "can_proceed should be boolean"
        assert isinstance(policy["requires_ack"], bool), "requires_ack should be boolean"
        assert policy["ack_code"] is None or isinstance(policy["ack_code"], str), \
            "ack_code should be None or string"
    
    def test_sunglasses_mutual_exclusivity(self):
        """Test: When sunglasses detected, GLASSES issue should NOT be emitted"""
        photo_path = TEST_PHOTOS_DIR / "inst_dark_tinted_lenses.webp"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_mutual_exclusivity", str(photo_path))
        
        issues = result.get("issues", [])
        issue_codes = [i["code"] for i in issues]
        
        # If SUNGLASSES is present, GLASSES should NOT be present
        if "SUNGLASSES" in issue_codes:
            assert "GLASSES" not in issue_codes, \
                "When SUNGLASSES is detected, GLASSES should NOT be emitted (mutual exclusivity)"
    
    def test_lens_metrics_present(self):
        """Test: New lens-specific metrics are present in result"""
        photo_path = TEST_PHOTOS_DIR / "inst_dark_tinted_lenses.webp"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_lens_metrics", str(photo_path))
        metrics = result.get("metrics", {})
        
        # Check new lens-specific metrics are present
        expected_metrics = [
            "frame_presence_score",
            "lens_dark_ratio",
            "lens_p10_luma",
            "lens_mean_luma",
            "iris_visibility_lens",
            "sunglasses_reason"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Expected metric '{metric}' not found in metrics"
    
    def test_sunglasses_requires_dark_lens_not_just_frame(self):
        """Test: SUNGLASSES fail requires dark lens, not just high edge density"""
        photo_path = TEST_PHOTOS_DIR / "inst_dark_tinted_lenses.webp"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_dark_lens_required", str(photo_path))
        metrics = result.get("metrics", {})
        issues = result.get("issues", [])
        
        sunglasses_issues = [i for i in issues if i["code"] == "SUNGLASSES" and i["severity"] == "fail"]
        
        if len(sunglasses_issues) > 0:
            # If sunglasses fail, lens should be dark
            lens_dark_ratio = metrics.get("lens_dark_ratio", 0.0)
            iris_visibility_lens = metrics.get("iris_visibility_lens", 1.0)
            
            # These should indicate actual lens tint, not just frame
            assert lens_dark_ratio > 0.2 or iris_visibility_lens < 0.5, \
                f"SUNGLASSES fail should be due to dark lens, not frame. " \
                f"lens_dark_ratio={lens_dark_ratio}, iris_visibility_lens={iris_visibility_lens}"


class TestIssuesSeverity:
    """Test that issues have correct severity levels"""
    
    def test_fail_issues_have_correct_severity(self):
        """Test: Blocking issues (SUNGLASSES, NO_FACE, etc.) have severity=fail"""
        blocking_codes = ["SUNGLASSES", "NO_FACE", "MULTIPLE_FACES", "BLURRED", 
                        "TOO_DARK", "TOO_BRIGHT", "HEAD_POSE", "EYES_CLOSED", 
                        "NOT_LOOKING_FORWARD", "HAT", "HAIR_OVER_EYES", 
                        "LOW_RESOLUTION", "PIXELATED"]
        
        photo_path = TEST_PHOTOS_DIR / "inst_dark_tinted_lenses.webp"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_severity", str(photo_path))
        issues = result.get("issues", [])
        
        for issue in issues:
            if issue["code"] in blocking_codes and issue["code"] != "GLASSES":
                # All blocking issues except GLASSES should be severity=fail
                assert issue["severity"] == "fail", \
                    f"{issue['code']} should have severity=fail, got {issue['severity']}"
    
    def test_glasses_is_warn_not_fail(self):
        """Test: GLASSES should have severity=warn, not fail
        
        POLICY: Normal glasses = WARNING only, user CAN proceed
        """
        # This test requires a photo with clear glasses (not sunglasses)
        # Skip if no such test image exists
        glasses_photo = TEST_PHOTOS_DIR / "inst_glasses.webp"
        
        if not glasses_photo.exists():
            # Also try alternate names
            for alt_name in ["glasses.jpg", "glasses.png", "clear_glasses.jpg", "clear_glasses.webp"]:
                glasses_photo = TEST_PHOTOS_DIR / alt_name
                if glasses_photo.exists():
                    break
            else:
                pytest.skip("No clear glasses test photo found")
        
        result = analyze_image_v2("test_glasses_warn", str(glasses_photo))
        issues = result.get("issues", [])
        final_status = result.get("final_status", "pass")
        
        glasses_issues = [i for i in issues if i["code"] == "GLASSES"]
        
        if len(glasses_issues) > 0:
            assert glasses_issues[0]["severity"] == "warn", \
                f"GLASSES should have severity=warn, got {glasses_issues[0]['severity']}"
            
            # final_status should be "warn" (not "fail")
            assert final_status == "warn", \
                f"final_status should be 'warn' for GLASSES, got '{final_status}'"
            
            # Check policy - user CAN proceed with GLASSES
            policy = result.get("policy", {})
            assert policy.get("can_proceed") == True, \
                "can_proceed should be True for GLASSES (non-blocking)"
            # No acknowledgement required for warnings
            assert policy.get("requires_ack") == False, \
                "requires_ack should be False (warnings don't require ack)"
            # warn_codes should contain GLASSES
            assert "GLASSES" in policy.get("warn_codes", []), \
                "warn_codes should contain 'GLASSES'"
    
    def test_clear_glasses_not_sunglasses_fail(self):
        """
        CRITICAL TEST: Clear glasses should NEVER produce SUNGLASSES fail.
        
        DECISION LOGIC:
        - SUNGLASSES: sunglasses_score > 0.7 AND iris_visibility < 0.35 => FAIL
        - GLASSES: glasses_score > 0.6 => WARN (user CAN proceed)
        - Otherwise: PASS
        """
        # Try to find a clear glasses test image
        glasses_photo = None
        for name in ["inst_glasses.webp", "glasses.jpg", "glasses.png", "clear_glasses.jpg", 
                     "clear_glasses.webp", "normal_glasses.jpg"]:
            path = TEST_PHOTOS_DIR / name
            if path.exists():
                glasses_photo = path
                break
        
        if glasses_photo is None:
            pytest.skip("No clear glasses test photo found")
        
        result = analyze_image_v2("test_clear_glasses_not_sunglasses", str(glasses_photo))
        issues = result.get("issues", [])
        metrics = result.get("metrics", {})
        final_status = result.get("final_status", "pass")
        
        # Clear glasses should NOT produce SUNGLASSES fail
        sunglasses_fail = [i for i in issues if i["code"] == "SUNGLASSES" and i["severity"] == "fail"]
        
        if len(sunglasses_fail) > 0:
            sunglasses_score = metrics.get("sunglasses_score", 0.0)
            iris_visibility = metrics.get("iris_visibility_lens", 1.0)
            
            # This should fail the test - clear glasses shouldn't trigger sunglasses
            assert False, \
                f"Clear glasses incorrectly classified as SUNGLASSES fail! " \
                f"sunglasses_score={sunglasses_score}, iris_visibility={iris_visibility}"
        
        # If GLASSES warn is emitted, that's correct behavior
        glasses_warn = [i for i in issues if i["code"] == "GLASSES" and i["severity"] == "warn"]
        if len(glasses_warn) > 0:
            # This is the expected behavior for clear glasses
            assert final_status == "warn", f"final_status should be 'warn', got '{final_status}'"
            
            policy = result.get("policy", {})
            assert policy.get("can_proceed") == True, "Clear glasses should allow proceeding"
            assert policy.get("requires_ack") == False, "Warnings don't require acknowledgement"
            assert "GLASSES" in policy.get("warn_codes", []), "warn_codes should contain GLASSES"


class TestGlassesDecisionLogic:
    """
    Test the new simplified glasses detection logic.
    
    DECISION LOGIC:
    - IF sunglasses_score > 0.7 AND iris_visibility < 0.35 => SUNGLASSES FAIL
    - ELSE IF glasses_score > 0.6 => GLASSES WARN (user CAN proceed)
    - ELSE => PASS
    """
    
    def test_glasses_score_triggers_warn(self):
        """
        Test: When glasses_score > 0.6, issues should contain GLASSES with severity=warn.
        User should be able to proceed (can_proceed=True).
        """
        # Try to find any photo that might have glasses
        glasses_photo = None
        for name in ["inst_glasses.webp", "glasses.jpg", "glasses.png", "clear_glasses.jpg"]:
            path = TEST_PHOTOS_DIR / name
            if path.exists():
                glasses_photo = path
                break
        
        if glasses_photo is None:
            pytest.skip("No glasses test photo found")
        
        result = analyze_image_v2("test_glasses_score", str(glasses_photo))
        metrics = result.get("metrics", {})
        issues = result.get("issues", [])
        policy = result.get("policy", {})
        final_status = result.get("final_status", "pass")
        
        glasses_score = metrics.get("glasses_score", 0.0)
        
        # If glasses_score > 0.6, GLASSES warn should be emitted
        if glasses_score > 0.6:
            glasses_issues = [i for i in issues if i["code"] == "GLASSES"]
            assert len(glasses_issues) > 0, \
                f"glasses_score={glasses_score} > 0.6 should trigger GLASSES warn"
            assert glasses_issues[0]["severity"] == "warn", \
                f"GLASSES should have severity=warn, got {glasses_issues[0]['severity']}"
            
            # User CAN proceed with warnings
            assert policy.get("can_proceed") == True, \
                "can_proceed should be True for GLASSES (user CAN continue)"
            assert final_status == "warn", \
                f"final_status should be 'warn', got '{final_status}'"
    
    def test_no_glasses_no_issue(self):
        """
        Test: Photo without glasses should have no GLASSES or SUNGLASSES issues.
        final_status should be "pass".
        """
        photo_path = TEST_PHOTOS_DIR / "goodphoto.jpg"
        
        if not photo_path.exists():
            pytest.skip(f"Test photo not found: {photo_path}")
        
        result = analyze_image_v2("test_no_glasses", str(photo_path))
        issues = result.get("issues", [])
        final_status = result.get("final_status", "pass")
        
        glasses_issues = [i for i in issues if i["code"] in ["GLASSES", "SUNGLASSES"]]
        
        # Good photo without glasses should have no glasses-related issues
        assert len(glasses_issues) == 0, \
            f"Good photo should have no glasses issues. Found: {[i['code'] for i in glasses_issues]}"
        
        # If no other issues, final_status should be "pass"
        if len(issues) == 0:
            assert final_status == "pass", \
                f"final_status should be 'pass' for good photo, got '{final_status}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

