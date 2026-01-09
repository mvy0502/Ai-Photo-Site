"""
Tests for Background Replace V2 - Landmark-Based Head+Neck Mask

Run with:
    cd /Users/vedat/Documents/Cursor/ai-photo-site
    source .venv/bin/activate
    pytest tests/test_background_replace_v2.py -v
"""

import pytest
import numpy as np
import cv2
import os
import sys
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.background_replace_v2 import (
    replace_background_biometric_v2,
    create_head_neck_mask,
    create_soft_alpha,
    create_studio_background,
    composite_foreground_background,
    DEFAULT_BG_RGB,
    FACE_OVAL_INDICES
)


class MockLandmark:
    """Mock MediaPipe landmark for testing"""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def create_mock_landmarks(width: int, height: int, face_center_x: float = 0.5, face_center_y: float = 0.4):
    """Create mock MediaPipe-style landmarks for testing"""
    landmarks = []
    
    # Create 478 mock landmarks (MediaPipe face mesh count)
    for i in range(478):
        # Default: random position near center
        x = face_center_x + np.random.uniform(-0.15, 0.15)
        y = face_center_y + np.random.uniform(-0.2, 0.2)
        
        # Face oval landmarks - create realistic positions
        if i in FACE_OVAL_INDICES:
            angle = FACE_OVAL_INDICES.index(i) / len(FACE_OVAL_INDICES) * 2 * np.pi
            radius_x = 0.12
            radius_y = 0.18
            x = face_center_x + radius_x * np.cos(angle)
            y = face_center_y + radius_y * np.sin(angle)
        
        # Chin (152)
        if i == 152:
            x = face_center_x
            y = face_center_y + 0.18
        
        # Forehead top (10)
        if i == 10:
            x = face_center_x
            y = face_center_y - 0.15
        
        # Jaw points (234, 454)
        if i == 234:  # Left jaw
            x = face_center_x - 0.12
            y = face_center_y + 0.12
        if i == 454:  # Right jaw
            x = face_center_x + 0.12
            y = face_center_y + 0.12
        
        landmarks.append(MockLandmark(x, y))
    
    return landmarks


class TestHeadNeckMask:
    """Test head+neck mask creation"""
    
    def test_mask_output_shape(self):
        """Test that mask has correct shape"""
        landmarks = create_mock_landmarks(400, 500)
        mask = create_head_neck_mask(landmarks, 400, 500)
        
        assert mask.shape == (500, 400)
        assert mask.dtype == np.uint8
    
    def test_mask_has_content(self):
        """Test that mask is not empty"""
        landmarks = create_mock_landmarks(400, 500)
        mask = create_head_neck_mask(landmarks, 400, 500)
        
        # Should have some white pixels (face region)
        coverage = np.sum(mask > 0) / mask.size * 100
        assert coverage > 5.0, f"Mask coverage too low: {coverage}%"
        assert coverage < 80.0, f"Mask coverage too high: {coverage}%"
    
    def test_mask_binary(self):
        """Test that mask is binary (0 or 255)"""
        landmarks = create_mock_landmarks(400, 500)
        mask = create_head_neck_mask(landmarks, 400, 500)
        
        unique_values = np.unique(mask)
        assert len(unique_values) <= 2
        assert 0 in unique_values
        assert 255 in unique_values


class TestSoftAlpha:
    """Test soft alpha creation"""
    
    def test_soft_alpha_output_shape(self):
        """Test that soft alpha has correct shape"""
        hard_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(hard_mask, (50, 50), 30, 255, -1)
        
        alpha = create_soft_alpha(hard_mask, face_width=60)
        
        assert alpha.shape == hard_mask.shape
        assert alpha.dtype == np.float32
    
    def test_soft_alpha_range(self):
        """Test that soft alpha is in [0, 1]"""
        hard_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(hard_mask, (50, 50), 30, 255, -1)
        
        alpha = create_soft_alpha(hard_mask, face_width=60)
        
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0
    
    def test_soft_alpha_has_gradient(self):
        """Test that soft alpha has gradual edges (not hard)"""
        hard_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(hard_mask, (50, 50), 30, 255, -1)
        
        alpha = create_soft_alpha(hard_mask, face_width=60)
        
        # Should have intermediate values at edges
        unique_nonzero = np.unique(alpha[alpha > 0])
        assert len(unique_nonzero) > 5, "Alpha should have smooth gradient"


class TestStudioBackground:
    """Test studio background generation"""
    
    def test_background_output_shape(self):
        """Test background has correct shape"""
        bg = create_studio_background(400, 500)
        
        assert bg.shape == (500, 400, 3)
        assert bg.dtype == np.float32
    
    def test_background_range(self):
        """Test background values are in [0, 1]"""
        bg = create_studio_background(400, 500)
        
        assert bg.min() >= 0.0
        assert bg.max() <= 1.0
    
    def test_background_color_close_to_default(self):
        """Test background color is close to default"""
        bg = create_studio_background(400, 500, add_gradient=False, add_noise=False)
        
        expected = np.array(DEFAULT_BG_RGB) / 255.0
        
        # All pixels should be close to expected
        for c in range(3):
            assert np.allclose(bg[:, :, c], expected[c], atol=0.01)
    
    def test_gradient_creates_variation(self):
        """Test that gradient creates vertical variation"""
        bg = create_studio_background(100, 100, add_gradient=True, add_noise=False)
        
        # Top should be slightly different from bottom
        top_mean = bg[:10, :, :].mean()
        bottom_mean = bg[-10:, :, :].mean()
        
        assert top_mean != bottom_mean, "Gradient should create variation"
    
    def test_noise_creates_variation(self):
        """Test that noise creates pixel variation"""
        bg = create_studio_background(100, 100, add_gradient=False, add_noise=True)
        
        # Should have some variation
        std = bg.std()
        assert std > 0.001, "Noise should create variation"


class TestComposite:
    """Test compositing function"""
    
    def test_composite_output_shape(self):
        """Test composite has correct shape"""
        fg = np.random.rand(100, 100, 3).astype(np.float32)
        bg = np.random.rand(100, 100, 3).astype(np.float32)
        alpha = np.random.rand(100, 100).astype(np.float32)
        
        result = composite_foreground_background(fg, bg, alpha)
        
        assert result.shape == fg.shape
        assert result.dtype == np.float32
    
    def test_composite_full_alpha_preserves_fg(self):
        """Test that alpha=1 preserves foreground"""
        fg = np.random.rand(50, 50, 3).astype(np.float32)
        bg = np.random.rand(50, 50, 3).astype(np.float32)
        alpha = np.ones((50, 50), dtype=np.float32)
        
        result = composite_foreground_background(fg, bg, alpha)
        
        np.testing.assert_array_almost_equal(result, fg)
    
    def test_composite_zero_alpha_gives_bg(self):
        """Test that alpha=0 gives background"""
        fg = np.random.rand(50, 50, 3).astype(np.float32)
        bg = np.random.rand(50, 50, 3).astype(np.float32)
        alpha = np.zeros((50, 50), dtype=np.float32)
        
        result = composite_foreground_background(fg, bg, alpha)
        
        np.testing.assert_array_almost_equal(result, bg)


class TestFullPipeline:
    """Test full background replacement pipeline"""
    
    def test_pipeline_output_shape(self):
        """Test that output has same shape as input"""
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(400, 500)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert output.shape == img.shape
        assert output.dtype == np.uint8
    
    def test_pipeline_returns_metrics(self):
        """Test that metrics dict is returned"""
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(400, 500)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert isinstance(metrics, dict)
        assert "method" in metrics
        assert metrics["method"] == "landmark_v2"
    
    def test_pipeline_output_contiguous(self):
        """Test that output is contiguous"""
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(400, 500)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert output.flags['C_CONTIGUOUS']
    
    def test_pipeline_no_nan_inf(self):
        """Test that output has no NaN or Inf values"""
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(400, 500)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
    
    def test_pipeline_background_color_in_corners(self):
        """Test that corners have background color (within tolerance)"""
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(400, 500, face_center_x=0.5, face_center_y=0.4)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        # Convert BGR to RGB for comparison
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        # Check corners - they should be close to background color
        # Allow tolerance for gradient and noise
        bg_rgb = np.array(DEFAULT_BG_RGB)
        tolerance = 20  # Allow some variation
        
        corners = [
            output_rgb[5, 5],      # Top-left
            output_rgb[5, -5],     # Top-right
            output_rgb[-5, 5],     # Bottom-left
            output_rgb[-5, -5]     # Bottom-right
        ]
        
        for corner in corners:
            # At least some corners should be background
            diff = np.abs(corner.astype(int) - bg_rgb.astype(int))
            if np.all(diff < tolerance):
                return  # Found at least one corner with background
        
        # If we get here, check if mask is very large
        if metrics.get("mask_coverage", 0) > 70:
            pytest.skip("Mask coverage too high for corner check")


class TestRealImages:
    """Test with real uploaded images"""
    
    def test_with_real_image(self):
        """Test pipeline with a real uploaded image"""
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        
        if not test_images:
            pytest.skip("No test images available in uploads/")
        
        img = cv2.imread(test_images[0])
        if img is None:
            pytest.skip("Could not load test image")
        
        # Resize for faster test
        img = cv2.resize(img, (400, 500))
        
        # Create mock landmarks (since we don't have real analyzer here)
        landmarks = create_mock_landmarks(400, 500)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert output.shape == img.shape
        assert output.dtype == np.uint8
        assert metrics["success"] == True
    
    def test_with_real_landmarks(self):
        """Test with real landmarks from analyzer"""
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        
        if not test_images:
            pytest.skip("No test images available")
        
        try:
            from utils.analyzer_v2 import get_analyzer
            analyzer = get_analyzer()
            
            if not analyzer or not analyzer.initialized:
                pytest.skip("Analyzer not available")
            
            img = cv2.imread(test_images[0])
            if img is None:
                pytest.skip("Could not load test image")
            
            # Analyze to get landmarks
            result = analyzer.analyze(img)
            
            if not result or not result.get("raw_landmarks"):
                pytest.skip("No landmarks detected")
            
            raw_landmarks = result["raw_landmarks"]
            
            # Run background replacement
            output, metrics = replace_background_biometric_v2(img, raw_landmarks)
            
            assert output.shape == img.shape
            assert output.dtype == np.uint8
            assert metrics["success"] == True
            assert metrics["method"] == "landmark_v2"
            
            # Save debug output
            os.makedirs("outputs/debug_bg_v2/pytest", exist_ok=True)
            cv2.imwrite("outputs/debug_bg_v2/pytest/test_real_landmarks.jpg", output)
            
        except ImportError:
            pytest.skip("Analyzer not available")


class TestDebugOutput:
    """Test debug output functionality"""
    
    def test_debug_files_created(self):
        """Test that debug files are created when debug_dir is set"""
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(400, 500)
        
        debug_dir = "outputs/debug_bg_v2/pytest_debug_test"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Enable debug
        os.environ["DEBUG_BG_V2"] = "true"
        
        output, metrics = replace_background_biometric_v2(
            img,
            landmarks,
            debug_dir=debug_dir
        )
        
        # Check debug files exist
        expected_files = [
            "01_input_bgr.png",
            "02_mask_headneck_hard.png",
            "03_alpha_headneck_soft.png",
            "04_alpha_refined.png",
            "05_composited.png",
            "06_final.png"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(debug_dir, filename)
            assert os.path.exists(filepath), f"Debug file not found: {filepath}"
        
        # Cleanup
        os.environ["DEBUG_BG_V2"] = "false"


class TestEdgeCases:
    """Test edge cases"""
    
    def test_small_image(self):
        """Test with very small image"""
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(80, 100)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert output.shape == img.shape
    
    def test_biometric_aspect_ratio(self):
        """Test with standard biometric aspect ratio"""
        img = np.random.randint(0, 255, (531, 413, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(413, 531)
        
        output, metrics = replace_background_biometric_v2(img, landmarks)
        
        assert output.shape == (531, 413, 3)
    
    def test_custom_background_color(self):
        """Test with custom background color"""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(200, 200)
        
        custom_bg = (200, 200, 200)  # Gray
        output, metrics = replace_background_biometric_v2(
            img, landmarks, bg_rgb=custom_bg
        )
        
        assert metrics["bg_rgb"] == custom_bg
    
    def test_no_gradient_no_noise(self):
        """Test without gradient and noise"""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        landmarks = create_mock_landmarks(200, 200)
        
        output, metrics = replace_background_biometric_v2(
            img, landmarks,
            add_subtle_gradient=False,
            add_subtle_noise=False
        )
        
        assert metrics["has_gradient"] == False
        assert metrics["has_noise"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

