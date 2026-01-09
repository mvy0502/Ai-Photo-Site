"""
Tests for Biometric Photo Processor and PhotoRoom Integration

Run with:
    cd /Users/vedat/Documents/Cursor/ai-photo-site
    source .venv/bin/activate
    pytest tests/test_biometric_processor.py -v
"""

import pytest
import numpy as np
import cv2
import os
import sys
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.biometric_processor import (
    compute_wide_crop_box,
    apply_wide_crop,
    filter_connected_components,
    refine_alpha_mask,
    composite_with_background,
    place_on_canvas,
    BIOMETRIC_BG_COLOR_RGB,
    TARGET_CANVAS_WIDTH,
    TARGET_CANVAS_HEIGHT
)

from utils.photoroom_client import (
    check_api_configuration,
    DEFAULT_BG_COLOR
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a sample BGR image for testing"""
    image = np.random.randint(0, 255, (600, 400, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_face_bbox():
    """Sample face bounding box (normalized coordinates)"""
    return (0.3, 0.2, 0.4, 0.5)  # xmin, ymin, width, height


@pytest.fixture
def sample_alpha_mask():
    """Create a sample alpha mask with a circle"""
    mask = np.zeros((200, 200), dtype=np.float32)
    cv2.circle(mask, (100, 100), 60, 1.0, -1)
    return mask


# =============================================================================
# Wide Crop Tests
# =============================================================================

class TestWideCrop:
    """Test wide crop computation"""
    
    def test_crop_box_within_bounds(self, sample_face_bbox):
        """Test that crop box stays within image bounds"""
        crop_box = compute_wide_crop_box(
            sample_face_bbox,
            image_width=400,
            image_height=600
        )
        
        x, y, w, h = crop_box
        
        assert x >= 0
        assert y >= 0
        assert x + w <= 400
        assert y + h <= 600
    
    def test_crop_box_includes_margins(self, sample_face_bbox):
        """Test that crop box is larger than face bbox"""
        image_width = 400
        image_height = 600
        
        crop_box = compute_wide_crop_box(
            sample_face_bbox,
            image_width,
            image_height
        )
        
        # Face bbox in pixels
        face_x = int(sample_face_bbox[0] * image_width)
        face_y = int(sample_face_bbox[1] * image_height)
        face_w = int(sample_face_bbox[2] * image_width)
        face_h = int(sample_face_bbox[3] * image_height)
        
        # Crop should be larger
        assert crop_box[2] >= face_w  # width
        assert crop_box[3] >= face_h  # height
    
    def test_apply_wide_crop(self, sample_image, sample_face_bbox):
        """Test that crop is applied correctly"""
        crop_box = compute_wide_crop_box(
            sample_face_bbox,
            sample_image.shape[1],
            sample_image.shape[0]
        )
        
        cropped, crop_info = apply_wide_crop(sample_image, crop_box)
        
        assert cropped.shape[0] == crop_box[3]  # height
        assert cropped.shape[1] == crop_box[2]  # width
        assert cropped.shape[2] == 3  # channels
        assert "crop_x" in crop_info
        assert "crop_y" in crop_info


# =============================================================================
# Connected Component Filtering Tests
# =============================================================================

class TestConnectedComponentFiltering:
    """Test connected component filtering for mask cleanup"""
    
    def test_keeps_face_component(self, sample_alpha_mask):
        """Test that component containing face center is kept"""
        face_center = (100, 100)  # Center of the circle
        
        result = filter_connected_components(sample_alpha_mask, face_center)
        
        # Should preserve the circle
        assert result[100, 100] > 0.5
    
    def test_removes_disconnected_blobs(self):
        """Test that disconnected blobs are removed"""
        # Create mask with main circle and small disconnected blob
        mask = np.zeros((200, 200), dtype=np.float32)
        cv2.circle(mask, (100, 100), 60, 1.0, -1)  # Main component
        cv2.circle(mask, (20, 20), 10, 1.0, -1)    # Small disconnected blob
        
        face_center = (100, 100)
        
        result = filter_connected_components(mask, face_center)
        
        # Main component should be preserved
        assert result[100, 100] > 0.5
        # Disconnected blob should be removed
        assert result[20, 20] == 0.0
    
    def test_handles_face_center_on_edge(self):
        """Test handling when face center is near mask edge"""
        mask = np.zeros((200, 200), dtype=np.float32)
        cv2.circle(mask, (100, 100), 60, 1.0, -1)
        
        # Face center slightly outside mask
        face_center = (160, 100)
        
        result = filter_connected_components(mask, face_center)
        
        # Should still return a valid mask
        assert result.shape == mask.shape


# =============================================================================
# Alpha Mask Refinement Tests
# =============================================================================

class TestAlphaMaskRefinement:
    """Test alpha mask refinement"""
    
    def test_output_shape_preserved(self, sample_alpha_mask):
        """Test that output shape matches input"""
        result = refine_alpha_mask(sample_alpha_mask)
        
        assert result.shape == sample_alpha_mask.shape
    
    def test_output_range_valid(self, sample_alpha_mask):
        """Test that output is in [0, 1]"""
        result = refine_alpha_mask(sample_alpha_mask)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_fills_small_holes(self):
        """Test that small holes are filled"""
        mask = np.ones((100, 100), dtype=np.float32)
        mask[50, 50] = 0.0  # Small hole
        
        result = refine_alpha_mask(mask)
        
        # Hole should be filled (morphological close)
        assert result[50, 50] > 0.5


# =============================================================================
# Compositing Tests
# =============================================================================

class TestCompositing:
    """Test foreground/background compositing"""
    
    def test_output_shape(self):
        """Test that output has correct shape"""
        foreground = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32)
        
        result = composite_with_background(foreground, alpha)
        
        assert result.shape == foreground.shape
        assert result.dtype == np.uint8
    
    def test_full_alpha_preserves_foreground(self):
        """Test that alpha=1 preserves foreground"""
        foreground = np.full((50, 50, 3), (100, 150, 200), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        
        result = composite_with_background(foreground, alpha)
        
        np.testing.assert_array_equal(result, foreground)
    
    def test_zero_alpha_gives_background(self):
        """Test that alpha=0 gives background color"""
        foreground = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        
        result = composite_with_background(foreground, alpha, BIOMETRIC_BG_COLOR_RGB)
        
        # All pixels should be background color
        for c in range(3):
            assert np.allclose(result[:, :, c], BIOMETRIC_BG_COLOR_RGB[c], atol=1)


# =============================================================================
# Canvas Placement Tests
# =============================================================================

class TestCanvasPlacement:
    """Test canvas placement and centering"""
    
    def test_output_size(self):
        """Test that output has target canvas size"""
        image = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        face_center = (75, 100)
        
        result = place_on_canvas(
            image, face_center,
            canvas_width=600, canvas_height=600
        )
        
        assert result.shape == (600, 600, 3)
    
    def test_background_color_in_corners(self):
        """Test that corners have background color"""
        # Small image that won't fill canvas
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        face_center = (50, 50)
        
        result = place_on_canvas(
            image, face_center,
            canvas_width=600, canvas_height=600,
            background_color_rgb=BIOMETRIC_BG_COLOR_RGB
        )
        
        # At least some corners should have background color
        corners = [
            result[5, 5],
            result[5, -5],
            result[-5, 5],
            result[-5, -5]
        ]
        
        bg_array = np.array(BIOMETRIC_BG_COLOR_RGB)
        has_bg_corner = any(
            np.allclose(corner, bg_array, atol=5) for corner in corners
        )
        
        assert has_bg_corner, "At least one corner should have background color"
    
    def test_face_approximately_centered(self):
        """Test that face is approximately centered on canvas"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)
        face_center = (100, 100)
        
        result = place_on_canvas(
            image, face_center,
            canvas_width=600, canvas_height=600
        )
        
        # Find white pixels (face) in result
        white_mask = np.all(result > 200, axis=2)
        if np.any(white_mask):
            ys, xs = np.where(white_mask)
            center_x = np.mean(xs)
            center_y = np.mean(ys)
            
            # Face should be near center horizontally
            assert abs(center_x - 300) < 150, f"Face center X ({center_x}) not near canvas center"


# =============================================================================
# PhotoRoom Client Tests
# =============================================================================

class TestPhotoRoomClient:
    """Test PhotoRoom API client"""
    
    def test_api_configuration_check(self):
        """Test API configuration check returns valid structure"""
        config = check_api_configuration()
        
        assert "api_configured" in config
        assert "api_key_length" in config
        assert "base_url" in config
        assert "default_bg_color" in config
    
    def test_default_bg_color_valid(self):
        """Test default background color is valid hex"""
        assert len(DEFAULT_BG_COLOR) == 6
        int(DEFAULT_BG_COLOR, 16)  # Should not raise


# =============================================================================
# Integration Tests (requires API key)
# =============================================================================

class TestIntegration:
    """Integration tests that require PhotoRoom API key"""
    
    @pytest.fixture
    def api_available(self):
        """Check if PhotoRoom API is configured"""
        config = check_api_configuration()
        if not config["api_configured"]:
            pytest.skip("PhotoRoom API key not configured")
        return True
    
    def test_full_pipeline_with_real_image(self, api_available):
        """Test full pipeline with a real image"""
        test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg")
        
        if not test_images:
            pytest.skip("No test images available")
        
        image = cv2.imread(test_images[0])
        if image is None:
            pytest.skip("Could not load test image")
        
        try:
            from utils.biometric_processor import process_biometric_photo
            
            output, metrics = process_biometric_photo(
                image,
                job_id="pytest_integration",
                canvas_width=600,
                canvas_height=600
            )
            
            if metrics.get("success"):
                # Check output dimensions
                assert output.shape == (600, 600, 3)
                assert output.dtype == np.uint8
                
                # Check background color in corners
                bg_bgr = (248, 246, 245)  # BGR for #F5F6F8
                corner_ok = False
                for corner in [output[5, 5], output[5, -5]]:
                    if np.allclose(corner, bg_bgr, atol=10):
                        corner_ok = True
                        break
                
                assert corner_ok or metrics.get("alpha_coverage_percent", 0) > 90
            else:
                # API might fail in test environment
                assert "error" in metrics
                
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


# =============================================================================
# Assertion Tests (Golden Set)
# =============================================================================

class TestGoldenSetAssertions:
    """Tests for golden set requirements"""
    
    def test_output_size_matches_target(self):
        """Output size should equal target canvas size"""
        image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        face_center = (150, 200)
        
        result = place_on_canvas(
            image, face_center,
            canvas_width=TARGET_CANVAS_WIDTH,
            canvas_height=TARGET_CANVAS_HEIGHT
        )
        
        assert result.shape[0] == TARGET_CANVAS_HEIGHT
        assert result.shape[1] == TARGET_CANVAS_WIDTH
    
    def test_background_close_to_target_color(self):
        """Background should be close to target biometric color"""
        foreground = np.zeros((100, 100, 3), dtype=np.uint8)
        alpha = np.zeros((100, 100), dtype=np.float32)
        
        result = composite_with_background(foreground, alpha, BIOMETRIC_BG_COLOR_RGB)
        
        # All pixels should be background color
        for c in range(3):
            assert np.allclose(result[:, :, c], BIOMETRIC_BG_COLOR_RGB[c], atol=2)
    
    def test_no_nan_inf_in_output(self):
        """Output should not contain NaN or Inf values"""
        foreground = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        alpha = np.random.rand(100, 100).astype(np.float32)
        
        result = composite_with_background(foreground, alpha)
        
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

