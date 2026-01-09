"""
Tests for Background Normalization Module

Run with:
    cd /Users/vedat/Documents/Cursor/ai-photo-site
    source .venv/bin/activate
    pytest tests/test_background_normalizer.py -v
"""

import pytest
import numpy as np
import cv2
import os
import glob
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.background_normalizer import (
    replace_background_biometric,
    normalize_background,
    create_soft_alpha,
    refine_alpha_mask,
    composite_with_background,
    get_person_mask_grabcut,
    BIOMETRIC_BG_COLOR
)


class TestSoftAlpha:
    """Test soft alpha creation"""
    
    def test_soft_alpha_thresholds(self):
        """Test that soft alpha correctly applies thresholds"""
        # Create test mask
        mask = np.array([0.0, 0.15, 0.45, 0.75, 1.0], dtype=np.float32)
        
        alpha = create_soft_alpha(mask, t0=0.15, t1=0.75)
        
        # Below t0 should be 0
        assert alpha[0] == 0.0
        assert alpha[1] == pytest.approx(0.0, abs=0.01)  # at threshold
        
        # Above t1 should be 1
        assert alpha[4] == 1.0
        assert alpha[3] == pytest.approx(1.0, abs=0.01)  # at threshold
        
        # Middle should be interpolated
        assert 0.0 < alpha[2] < 1.0
    
    def test_soft_alpha_preserves_range(self):
        """Test that soft alpha output is in [0, 1]"""
        mask = np.random.rand(100, 100).astype(np.float32)
        alpha = create_soft_alpha(mask)
        
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0


class TestRefineAlpha:
    """Test alpha mask refinement"""
    
    def test_refine_alpha_output_shape(self):
        """Test that refined alpha has same shape"""
        mask = np.random.rand(100, 100).astype(np.float32)
        refined = refine_alpha_mask(mask)
        
        assert refined.shape == mask.shape
    
    def test_refine_alpha_output_range(self):
        """Test that refined alpha is in [0, 1]"""
        mask = np.random.rand(100, 100).astype(np.float32)
        refined = refine_alpha_mask(mask)
        
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0
    
    def test_refine_alpha_fills_holes(self):
        """Test that refinement fills small holes"""
        # Create mask with a small hole in center
        mask = np.ones((50, 50), dtype=np.float32)
        mask[24, 24] = 0.0  # single pixel hole
        
        refined = refine_alpha_mask(mask)
        
        # Hole should be filled (morphological close)
        assert refined[24, 24] > 0.5


class TestComposite:
    """Test compositing function"""
    
    def test_composite_output_shape(self):
        """Test that composite has correct shape"""
        fg = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32)
        
        result = composite_with_background(fg, alpha)
        
        assert result.shape == fg.shape
        assert result.dtype == np.uint8
    
    def test_composite_full_alpha_preserves_fg(self):
        """Test that alpha=1 preserves foreground"""
        fg = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        
        result = composite_with_background(fg, alpha)
        
        np.testing.assert_array_equal(result, fg)
    
    def test_composite_zero_alpha_gives_bg(self):
        """Test that alpha=0 gives background color"""
        fg = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        bg_color = (100, 150, 200)
        
        result = composite_with_background(fg, alpha, bg_color)
        
        expected = np.zeros_like(fg)
        expected[:, :, 0] = bg_color[0]
        expected[:, :, 1] = bg_color[1]
        expected[:, :, 2] = bg_color[2]
        
        np.testing.assert_array_equal(result, expected)
    
    def test_composite_biometric_bg_color(self):
        """Test default biometric background color"""
        fg = np.zeros((50, 50, 3), dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        
        result = composite_with_background(fg, alpha)  # default bg_color
        
        # Should be BIOMETRIC_BG_COLOR = (245, 246, 248) RGB
        assert result[0, 0, 0] == BIOMETRIC_BG_COLOR[0]  # R
        assert result[0, 0, 1] == BIOMETRIC_BG_COLOR[1]  # G
        assert result[0, 0, 2] == BIOMETRIC_BG_COLOR[2]  # B
    
    def test_composite_contiguous(self):
        """Test that output is contiguous"""
        fg = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        alpha = np.random.rand(50, 50).astype(np.float32)
        
        result = composite_with_background(fg, alpha)
        
        assert result.flags['C_CONTIGUOUS']


class TestGrabCutMask:
    """Test GrabCut segmentation fallback"""
    
    def test_grabcut_output_shape(self):
        """Test GrabCut returns correct shape"""
        # Create synthetic test image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add a "person" in center
        cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (180, 150, 130), -1)
        
        mask = get_person_mask_grabcut(img)
        
        if mask is not None:  # GrabCut might fail on synthetic images
            assert mask.shape == (200, 200)
            assert mask.dtype == np.float32
    
    def test_grabcut_output_range(self):
        """Test GrabCut mask is in [0, 1]"""
        # Use a real test image if available
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        
        if not test_images:
            pytest.skip("No test images available")
        
        img = cv2.imread(test_images[0])
        if img is None:
            pytest.skip("Could not load test image")
        
        # Resize for faster test
        img = cv2.resize(img, (200, 200))
        
        mask = get_person_mask_grabcut(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if mask is not None:
            assert mask.min() >= 0.0
            assert mask.max() <= 1.0


class TestReplaceBackground:
    """Test full background replacement pipeline"""
    
    def test_replace_background_output_shape(self):
        """Test that output has same shape as input"""
        # Create synthetic test image
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        output, metrics = replace_background_biometric(img, job_id="test_shape")
        
        assert output.shape == img.shape
        assert output.dtype == np.uint8
    
    def test_replace_background_returns_metrics(self):
        """Test that metrics dictionary is returned"""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        output, metrics = replace_background_biometric(img, job_id="test_metrics")
        
        assert isinstance(metrics, dict)
        assert "seg_model" in metrics
    
    def test_replace_background_contiguous(self):
        """Test that output is contiguous"""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        output, metrics = replace_background_biometric(img, job_id="test_contiguous")
        
        assert output.flags['C_CONTIGUOUS']
    
    def test_replace_background_with_real_image(self):
        """Test with a real uploaded image"""
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        
        if not test_images:
            pytest.skip("No test images available")
        
        img = cv2.imread(test_images[0])
        if img is None:
            pytest.skip("Could not load test image")
        
        output, metrics = replace_background_biometric(img, job_id="test_real")
        
        # Check output properties
        assert output.shape == img.shape
        assert output.dtype == np.uint8
        assert output.flags['C_CONTIGUOUS']
        
        # Check that background was applied (corners should be background color)
        # Note: This might not be exact due to feathering
        if metrics.get("success", False):
            # At least check that some processing happened
            assert metrics["mask_coverage_raw"] > 0
            assert metrics["mask_coverage_refined"] > 0


class TestLegacyWrapper:
    """Test legacy normalize_background wrapper"""
    
    def test_legacy_wrapper_works(self):
        """Test that legacy wrapper calls new function"""
        # Use a real image to ensure proper segmentation coverage
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        
        if test_images:
            img = cv2.imread(test_images[0])
            if img is not None:
                img = cv2.resize(img, (200, 200))
        else:
            # Create synthetic image with a "person" shape
            img = np.full((200, 200, 3), (200, 200, 200), dtype=np.uint8)
            cv2.ellipse(img, (100, 80), (50, 60), 0, 0, 360, (150, 130, 110), -1)
        
        output, metrics = normalize_background(
            img,
            None,  # landmarks_or_bbox ignored
            bg_color=(245, 246, 248),
            job_id="test_legacy"
        )
        
        assert output.shape == img.shape
        # seg_model_used is always present
        assert "seg_model_used" in metrics or "seg_model" in metrics
        # alpha_mean only present on success
        if metrics.get("success", False):
            assert "alpha_mean" in metrics


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_small_image(self):
        """Test with very small image"""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        output, metrics = replace_background_biometric(img, job_id="test_small")
        
        assert output.shape == img.shape
    
    def test_non_square_image(self):
        """Test with non-square aspect ratio"""
        img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        
        output, metrics = replace_background_biometric(img, job_id="test_nonsquare")
        
        assert output.shape == img.shape
    
    def test_biometric_aspect_ratio(self):
        """Test with standard biometric aspect ratio (413x531)"""
        img = np.random.randint(0, 255, (531, 413, 3), dtype=np.uint8)
        
        output, metrics = replace_background_biometric(img, job_id="test_biometric")
        
        assert output.shape == (531, 413, 3)


class TestDebugOutput:
    """Test debug output functionality"""
    
    def test_debug_files_created(self):
        """Test that debug files are created when DEBUG_BG_PIPELINE=true"""
        import os
        os.environ["DEBUG_BG_PIPELINE"] = "true"
        
        # Reload module to pick up env change
        import importlib
        import utils.background_normalizer as bg_module
        importlib.reload(bg_module)
        
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        if not test_images:
            pytest.skip("No test images available")
        
        img = cv2.imread(test_images[0])
        if img is None:
            pytest.skip("Could not load test image")
        
        # Run with unique job_id
        job_id = "pytest_debug_test"
        output, metrics = bg_module.replace_background_biometric(img, job_id=job_id)
        
        # Check debug files exist
        debug_dir = "outputs/debug_bg"
        expected_files = [
            f"{job_id}_01_input_bgr.png",
            f"{job_id}_02_mask_raw.png",
            f"{job_id}_03_alpha_soft.png",
            f"{job_id}_04_alpha_refined.png",
            f"{job_id}_05_composited.png",
            f"{job_id}_06_final_bgr.png",
        ]
        
        for filename in expected_files:
            filepath = os.path.join(debug_dir, filename)
            assert os.path.exists(filepath), f"Debug file not found: {filepath}"
        
        # Cleanup - reset env
        os.environ["DEBUG_BG_PIPELINE"] = "false"


class TestIntegration:
    """Integration tests with real workflow"""
    
    def test_full_pipeline_good_photo(self):
        """Test full pipeline with a good quality photo"""
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        if not test_images:
            pytest.skip("No test images available")
        
        img = cv2.imread(test_images[0])
        if img is None:
            pytest.skip("Could not load test image")
        
        # Process
        output, metrics = replace_background_biometric(img, job_id="integration_good")
        
        # Validate
        assert output is not None
        assert metrics.get("success", False) or "error" in metrics
        
        if metrics.get("success"):
            # Check corners are close to biometric gray
            expected_bgr = (248, 246, 245)  # BGR of #F5F6F8
            
            for corner in [(5, 5), (5, -5), (-5, 5), (-5, -5)]:
                pixel = output[corner[0], corner[1]]
                # Allow some tolerance due to anti-aliasing
                if not np.allclose(pixel, expected_bgr, atol=30):
                    # Might be part of the person, check coverage
                    if metrics["mask_coverage_refined"] > 95:
                        continue  # Very high coverage, corners might be person
                    pytest.fail(f"Corner pixel {corner} = {pixel}, expected ~{expected_bgr}")
    
    def test_different_background_colors(self):
        """Test that different background colors work"""
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        if not test_images:
            pytest.skip("No test images available")
        
        img = cv2.imread(test_images[0])
        if img is None:
            pytest.skip("Could not load test image")
        
        # Test with different colors
        colors = [
            (255, 255, 255),  # White
            (128, 128, 128),  # Gray
            (200, 200, 200),  # Light gray
        ]
        
        for bg_color in colors:
            output, metrics = replace_background_biometric(
                img.copy(), 
                bg_color=bg_color, 
                job_id=f"test_color_{bg_color[0]}"
            )
            
            assert output is not None
            assert output.shape == img.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
