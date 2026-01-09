"""
Background Replace V2 - Professional Landmark-Based Head+Neck Mask

Uses MediaPipe FaceLandmarker V2 landmarks to create a clean HEAD + NECK mask
that EXCLUDES shoulders/arms for passport-style photo output.

Key differences from V1 (person segmentation):
- No ugly shoulder/arm blobs
- Clean contour like professional passport photo
- Uses face oval landmarks + neck trapezoid
- Professional studio-like background with subtle gradient and noise

Author: AI Photo Site Team
Version: 2.0
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import time

# =============================================================================
# Constants
# =============================================================================

# MediaPipe Face Mesh: Face Oval / Silhouette Indices (36 points)
# These outline the face boundary from forehead through jawline
FACE_OVAL_INDICES = [
    10,   # Top center forehead
    338, 297, 332, 284, 251, 389, 356,  # Right forehead to temple
    454,  # Right temple
    323, 361, 288, 397, 365, 379, 378, 400, 377,  # Right cheek to jaw
    152,  # Chin center
    148, 176, 149, 150, 136, 172,  # Left jaw to cheek
    58,   # Left temple
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,  # Left temple to forehead
]

# Jawline landmarks (left to right for neck trapezoid)
JAW_LEFT_INDEX = 234   # Left jaw corner
JAW_RIGHT_INDEX = 454  # Right jaw corner
CHIN_INDEX = 152       # Chin bottom

# Forehead top for hair margin
FOREHEAD_TOP_INDEX = 10

# Default background color: biometric light gray #F5F6F8
DEFAULT_BG_RGB = (245, 246, 248)

# Hair margin as ratio of face width
HAIR_MARGIN_RATIO = 0.03  # 3% of face width for hair dilation

# Neck dimensions relative to face
NECK_WIDTH_RATIO = 0.55      # Neck width = 55% of jaw width
NECK_HEIGHT_RATIO = 0.25     # Neck extends 25% of face height below chin

# Feathering parameters
FEATHER_SIGMA_RATIO = 0.015  # Sigma = 1.5% of face width
ERODE_PX = 2                 # Erosion pixels for defringe

# Background gradient/noise
GRADIENT_DELTA = 4           # Top is this many values darker
NOISE_STD = 1.5              # Subtle noise standard deviation

# Debug
DEBUG_ENABLED = os.getenv("DEBUG_BG_V2", "false").lower() == "true"


# =============================================================================
# Utility Functions
# =============================================================================

def _debug_save(img: np.ndarray, path: str, force: bool = False):
    """Save debug image"""
    if not DEBUG_ENABLED and not force:
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert float to uint8 if needed
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1.0:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_save = img.copy()
    
    img_save = np.ascontiguousarray(img_save)
    cv2.imwrite(path, img_save)
    print(f"  [DEBUG V2] Saved: {path}")


def _validate_landmarks(landmarks_list: List) -> bool:
    """Validate that landmarks list has enough points"""
    if landmarks_list is None:
        return False
    if len(landmarks_list) < 468:  # MediaPipe Face Mesh has 468+ landmarks
        return False
    return True


def _get_landmark_point(landmarks_list: List, idx: int, w: int, h: int) -> Tuple[int, int]:
    """Get pixel coordinates for a landmark index"""
    lm = landmarks_list[idx]
    return (int(lm.x * w), int(lm.y * h))


def _get_face_bbox(landmarks_list: List, w: int, h: int) -> Tuple[int, int, int, int]:
    """Calculate face bounding box from landmarks"""
    xs = [landmarks_list[i].x * w for i in range(len(landmarks_list))]
    ys = [landmarks_list[i].y * h for i in range(len(landmarks_list))]
    
    x_min = int(min(xs))
    x_max = int(max(xs))
    y_min = int(min(ys))
    y_max = int(max(ys))
    
    face_width = x_max - x_min
    face_height = y_max - y_min
    
    return (x_min, y_min, face_width, face_height)


# =============================================================================
# Mask Generation
# =============================================================================

def create_head_neck_mask(
    landmarks_list: List,
    img_width: int,
    img_height: int,
    include_neck: bool = True,
    hair_margin_ratio: float = HAIR_MARGIN_RATIO,
    neck_width_ratio: float = NECK_WIDTH_RATIO,
    neck_height_ratio: float = NECK_HEIGHT_RATIO
) -> np.ndarray:
    """
    Create a hard mask for head + neck region from face landmarks.
    
    Args:
        landmarks_list: MediaPipe FaceLandmarker raw landmarks
        img_width: Image width in pixels
        img_height: Image height in pixels
        include_neck: Whether to include neck trapezoid
        hair_margin_ratio: Extra margin for hair (ratio of face width)
        neck_width_ratio: Neck width as ratio of jaw width
        neck_height_ratio: Neck height as ratio of face height
    
    Returns:
        Binary mask (uint8) with 255 for head/neck, 0 for background
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Get face oval points
    face_oval_points = []
    for idx in FACE_OVAL_INDICES:
        if idx < len(landmarks_list):
            pt = _get_landmark_point(landmarks_list, idx, img_width, img_height)
            face_oval_points.append(pt)
    
    if len(face_oval_points) < 20:
        print(f"  ⚠️ Not enough face oval points: {len(face_oval_points)}")
        return mask
    
    # Convert to numpy array
    face_oval_pts = np.array(face_oval_points, dtype=np.int32)
    
    # Fill face oval polygon
    cv2.fillPoly(mask, [face_oval_pts], 255)
    
    # Get face metrics for proportional calculations
    face_bbox = _get_face_bbox(landmarks_list, img_width, img_height)
    face_x, face_y, face_width, face_height = face_bbox
    
    # Add neck trapezoid if requested
    if include_neck:
        # Get jaw points
        jaw_left = _get_landmark_point(landmarks_list, JAW_LEFT_INDEX, img_width, img_height)
        jaw_right = _get_landmark_point(landmarks_list, JAW_RIGHT_INDEX, img_width, img_height)
        chin = _get_landmark_point(landmarks_list, CHIN_INDEX, img_width, img_height)
        
        # Jaw width
        jaw_width = abs(jaw_right[0] - jaw_left[0])
        
        # Neck trapezoid dimensions
        neck_width = int(jaw_width * neck_width_ratio)
        neck_height = int(face_height * neck_height_ratio)
        
        # Trapezoid corners
        # Top: slightly inset from jaw points, at chin level
        neck_top_left = (chin[0] - neck_width // 2, chin[1])
        neck_top_right = (chin[0] + neck_width // 2, chin[1])
        
        # Bottom: wider and below chin
        bottom_y = min(chin[1] + neck_height, img_height - 1)
        neck_bottom_width = int(neck_width * 1.1)  # Slightly wider at bottom
        neck_bottom_left = (chin[0] - neck_bottom_width // 2, bottom_y)
        neck_bottom_right = (chin[0] + neck_bottom_width // 2, bottom_y)
        
        # Create neck trapezoid
        neck_pts = np.array([
            neck_top_left,
            neck_top_right,
            neck_bottom_right,
            neck_bottom_left
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [neck_pts], 255)
    
    # Add hair margin via dilation
    if hair_margin_ratio > 0:
        hair_margin_px = max(3, int(face_width * hair_margin_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hair_margin_px * 2 + 1, hair_margin_px * 2 + 1))
        
        # Only dilate the top portion (hair area) more
        # Split mask into top (hair) and bottom (face/neck)
        hair_boundary = face_y + int(face_height * 0.3)  # Top 30% is hair region
        
        # Create hair region mask
        hair_mask = mask.copy()
        hair_mask[hair_boundary:, :] = 0
        
        # Dilate hair region more
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hair_margin_px * 3 + 1, hair_margin_px * 3 + 1))
        hair_mask_dilated = cv2.dilate(hair_mask, hair_kernel, iterations=1)
        
        # Dilate rest of mask normally
        face_mask = mask.copy()
        face_mask[:hair_boundary, :] = 0
        face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=1)
        
        # Combine
        mask = np.maximum(hair_mask_dilated, face_mask_dilated)
    
    return mask


def create_soft_alpha(
    hard_mask: np.ndarray,
    face_width: int,
    feather_sigma_ratio: float = FEATHER_SIGMA_RATIO,
    erode_px: int = ERODE_PX
) -> np.ndarray:
    """
    Create soft alpha mask with professional feathering and defringe.
    
    Args:
        hard_mask: Binary mask (uint8, 0 or 255)
        face_width: Face width for proportional sigma calculation
        feather_sigma_ratio: Feather sigma as ratio of face width
        erode_px: Pixels to erode for defringe
    
    Returns:
        Float32 alpha mask [0..1]
    """
    # Convert to float
    alpha = hard_mask.astype(np.float32) / 255.0
    
    # Calculate sigma based on face size
    sigma = max(3.0, face_width * feather_sigma_ratio)
    
    # Step 1: Initial Gaussian blur for soft edges
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma)
    
    # Step 2: Defringe - erode slightly to remove background halos
    if erode_px > 0:
        # Create eroded version
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        alpha_eroded = cv2.erode(alpha_uint8, kernel, iterations=1)
        alpha = alpha_eroded.astype(np.float32) / 255.0
        
        # Step 3: Final light blur after erosion
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma * 0.5)
    
    # Clamp to [0, 1]
    alpha = np.clip(alpha, 0.0, 1.0)
    
    return alpha


# =============================================================================
# Background Generation
# =============================================================================

def create_studio_background(
    width: int,
    height: int,
    bg_rgb: Tuple[int, int, int] = DEFAULT_BG_RGB,
    add_gradient: bool = True,
    gradient_delta: int = GRADIENT_DELTA,
    add_noise: bool = True,
    noise_std: float = NOISE_STD
) -> np.ndarray:
    """
    Create professional studio-like background.
    
    Args:
        width: Image width
        height: Image height
        bg_rgb: Base RGB color
        add_gradient: Add subtle vertical gradient (top darker)
        gradient_delta: How much darker the top is
        add_noise: Add subtle noise texture
        noise_std: Noise standard deviation
    
    Returns:
        RGB float32 background [0..1]
    """
    # Create base background (float32, 0..1)
    bg = np.ones((height, width, 3), dtype=np.float32)
    bg[:, :, 0] = bg_rgb[0] / 255.0
    bg[:, :, 1] = bg_rgb[1] / 255.0
    bg[:, :, 2] = bg_rgb[2] / 255.0
    
    # Add subtle vertical gradient (top is slightly darker)
    if add_gradient:
        gradient = np.linspace(0, 1, height).reshape(-1, 1, 1)
        gradient_effect = gradient * (gradient_delta / 255.0)
        bg = bg - (1 - gradient) * (gradient_delta / 255.0)  # Top darker
    
    # Add subtle noise texture
    if add_noise:
        noise = np.random.normal(0, noise_std / 255.0, bg.shape).astype(np.float32)
        bg = bg + noise
    
    # Clamp to valid range and ensure float32
    bg = np.clip(bg, 0.0, 1.0).astype(np.float32)
    
    return bg


# =============================================================================
# Compositing
# =============================================================================

def composite_foreground_background(
    fg_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    alpha: np.ndarray
) -> np.ndarray:
    """
    Composite foreground with background using alpha mask.
    
    All inputs must be float32 [0..1].
    
    Args:
        fg_rgb: Foreground RGB [0..1]
        bg_rgb: Background RGB [0..1]
        alpha: Alpha mask (H, W) [0..1]
    
    Returns:
        Composited RGB float32 [0..1]
    """
    # Ensure alpha is 3D for broadcasting
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
    
    # Composite: out = fg * alpha + bg * (1 - alpha)
    out = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    
    return out


# =============================================================================
# Main Function
# =============================================================================

def replace_background_biometric_v2(
    bgr_image: np.ndarray,
    face_landmarks: List,
    *,
    bg_rgb: Tuple[int, int, int] = DEFAULT_BG_RGB,
    add_subtle_gradient: bool = True,
    add_subtle_noise: bool = True,
    debug_dir: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Replace background with biometric studio-like background using landmark-based mask.
    
    This function creates a clean HEAD + NECK mask that EXCLUDES shoulders/arms,
    producing passport-style photo output without ugly artifacts.
    
    Args:
        bgr_image: Input BGR image (from cv2.imread)
        face_landmarks: MediaPipe FaceLandmarker raw landmarks list
        bg_rgb: Background RGB color (default: #F5F6F8)
        add_subtle_gradient: Add subtle vertical gradient
        add_subtle_noise: Add subtle noise texture
        debug_dir: Directory to save debug images (None to disable)
    
    Returns:
        (output_bgr, metrics_dict)
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"[BG_REPLACE_V2] Starting landmark-based background replacement")
    print(f"  Target BG: RGB{bg_rgb} = #{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}")
    print(f"{'='*60}")
    
    # Validate inputs
    if bgr_image is None:
        raise ValueError("Input image is None")
    
    h, w = bgr_image.shape[:2]
    print(f"\n[INPUT] Size: {w}x{h}, dtype: {bgr_image.dtype}")
    
    # Validate landmarks
    if not _validate_landmarks(face_landmarks):
        print("  ⚠️ Invalid or missing landmarks, cannot proceed")
        return bgr_image, {
            "success": False,
            "error": "invalid_landmarks",
            "method": "none"
        }
    
    # Make contiguous copy
    bgr_image = np.ascontiguousarray(bgr_image.copy())
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Debug: Save input (force=True when debug_dir is provided)
    if debug_dir:
        _debug_save(bgr_image, os.path.join(debug_dir, "01_input_bgr.png"), force=True)
    
    # ==========================================================================
    # Step 1: Create head+neck hard mask from landmarks
    # ==========================================================================
    print("\n[STEP 1] Creating head+neck mask from landmarks...")
    
    face_bbox = _get_face_bbox(face_landmarks, w, h)
    face_x, face_y, face_width, face_height = face_bbox
    print(f"  Face bbox: x={face_x}, y={face_y}, w={face_width}, h={face_height}")
    
    hard_mask = create_head_neck_mask(
        face_landmarks,
        w, h,
        include_neck=True,
        hair_margin_ratio=HAIR_MARGIN_RATIO,
        neck_width_ratio=NECK_WIDTH_RATIO,
        neck_height_ratio=NECK_HEIGHT_RATIO
    )
    
    mask_coverage = np.sum(hard_mask > 0) / hard_mask.size * 100
    print(f"  Hard mask coverage: {mask_coverage:.1f}%")
    
    if debug_dir:
        _debug_save(hard_mask, os.path.join(debug_dir, "02_mask_headneck_hard.png"), force=True)
    
    # Validate mask coverage
    if mask_coverage < 3.0:
        print(f"  ⚠️ Mask coverage too low ({mask_coverage:.1f}%), returning original")
        return bgr_image, {
            "success": False,
            "error": "mask_coverage_too_low",
            "coverage": mask_coverage,
            "method": "landmark_v2"
        }
    
    # ==========================================================================
    # Step 2: Create soft alpha with feathering
    # ==========================================================================
    print("\n[STEP 2] Creating soft alpha with feathering...")
    
    soft_alpha = create_soft_alpha(
        hard_mask,
        face_width,
        feather_sigma_ratio=FEATHER_SIGMA_RATIO,
        erode_px=ERODE_PX
    )
    
    alpha_mean = float(soft_alpha.mean())
    print(f"  Alpha mean: {alpha_mean:.3f}")
    
    if debug_dir:
        _debug_save((soft_alpha * 255).astype(np.uint8), os.path.join(debug_dir, "03_alpha_headneck_soft.png"), force=True)
    
    # ==========================================================================
    # Step 3: Refine alpha (additional edge cleanup)
    # ==========================================================================
    print("\n[STEP 3] Refining alpha...")
    
    # Apply bilateral filter to smooth while preserving edges
    alpha_uint8 = (soft_alpha * 255).astype(np.uint8)
    alpha_refined_uint8 = cv2.bilateralFilter(alpha_uint8, 9, 75, 75)
    refined_alpha = alpha_refined_uint8.astype(np.float32) / 255.0
    
    # Final clip
    refined_alpha = np.clip(refined_alpha, 0.0, 1.0)
    
    if debug_dir:
        _debug_save((refined_alpha * 255).astype(np.uint8), os.path.join(debug_dir, "04_alpha_refined.png"), force=True)
    
    # ==========================================================================
    # Step 4: Create studio background
    # ==========================================================================
    print("\n[STEP 4] Creating studio background...")
    
    bg_rgb_float = create_studio_background(
        w, h,
        bg_rgb=bg_rgb,
        add_gradient=add_subtle_gradient,
        gradient_delta=GRADIENT_DELTA,
        add_noise=add_subtle_noise,
        noise_std=NOISE_STD
    )
    
    print(f"  Background: gradient={add_subtle_gradient}, noise={add_subtle_noise}")
    
    # ==========================================================================
    # Step 5: Composite
    # ==========================================================================
    print("\n[STEP 5] Compositing...")
    
    # Convert foreground to float32 [0..1]
    fg_rgb_float = rgb_image.astype(np.float32) / 255.0
    
    # Composite
    composited_rgb = composite_foreground_background(fg_rgb_float, bg_rgb_float, refined_alpha)
    
    # Check for NaN/Inf
    if np.any(np.isnan(composited_rgb)) or np.any(np.isinf(composited_rgb)):
        raise RuntimeError("Composited image contains NaN or Inf values")
    
    if debug_dir:
        composited_bgr = cv2.cvtColor((composited_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _debug_save(composited_bgr, os.path.join(debug_dir, "05_composited.png"), force=True)
    
    # ==========================================================================
    # Step 6: Convert back to BGR uint8
    # ==========================================================================
    print("\n[STEP 6] Final conversion...")
    
    # Convert to uint8
    output_rgb = np.clip(composited_rgb * 255.0, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    
    # Ensure contiguous
    output_bgr = np.ascontiguousarray(output_bgr)
    
    # Final validation
    assert output_bgr.shape == (h, w, 3), f"Output shape mismatch: {output_bgr.shape}"
    assert output_bgr.dtype == np.uint8, f"Output dtype: {output_bgr.dtype}"
    assert output_bgr.flags['C_CONTIGUOUS'], "Output not contiguous"
    
    if debug_dir:
        _debug_save(output_bgr, os.path.join(debug_dir, "06_final.png"), force=True)
    
    # ==========================================================================
    # Metrics
    # ==========================================================================
    elapsed = time.time() - start_time
    
    metrics = {
        "success": True,
        "method": "landmark_v2",
        "face_width": face_width,
        "face_height": face_height,
        "mask_coverage": round(mask_coverage, 1),
        "alpha_mean": round(alpha_mean, 3),
        "bg_rgb": bg_rgb,
        "bg_hex": f"#{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}",
        "has_gradient": add_subtle_gradient,
        "has_noise": add_subtle_noise,
        "processing_time_ms": round(elapsed * 1000, 1)
    }
    
    print(f"\n{'='*60}")
    print(f"[BG_REPLACE_V2] Complete in {elapsed*1000:.0f}ms")
    print(f"  Method: landmark_v2 (head+neck mask)")
    print(f"  Coverage: {mask_coverage:.1f}%")
    print(f"  Output: {output_bgr.shape}, {output_bgr.dtype}")
    print(f"{'='*60}\n")
    
    return output_bgr, metrics


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    import sys
    import glob
    
    # Enable debug
    os.environ["DEBUG_BG_V2"] = "true"
    
    print("="*60)
    print("Background Replace V2 - Standalone Test")
    print("="*60)
    
    # Try to use analyzer to get landmarks
    try:
        from analyzer_v2 import get_analyzer
        analyzer = get_analyzer()
        
        if not analyzer or not analyzer.initialized:
            print("Analyzer not available")
            sys.exit(1)
        
        # Find test image
        test_images = glob.glob("../uploads/*.jpg") + glob.glob("../uploads/*.jpeg")
        if not test_images:
            print("No test images found")
            sys.exit(1)
        
        test_path = test_images[0]
        print(f"\nTesting with: {test_path}")
        
        # Load and analyze
        img_bgr = cv2.imread(test_path)
        if img_bgr is None:
            print(f"Failed to load {test_path}")
            sys.exit(1)
        
        # Get landmarks
        result = analyzer.analyze(img_bgr)
        if not result or not result.get("face_detected"):
            print("No face detected")
            sys.exit(1)
        
        # Need raw landmarks - re-detect
        from mediapipe.tasks.python import vision
        from mediapipe import Image as MPImage
        
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=vision.ImageFormat.SRGB, data=rgb)
        detection = analyzer.landmarker.detect(mp_image)
        
        if not detection.face_landmarks:
            print("No landmarks found")
            sys.exit(1)
        
        raw_landmarks = detection.face_landmarks[0]
        
        # Test background replacement
        debug_dir = "../outputs/debug_bg_v2"
        os.makedirs(debug_dir, exist_ok=True)
        
        output, metrics = replace_background_biometric_v2(
            img_bgr,
            raw_landmarks,
            bg_rgb=(245, 246, 248),
            add_subtle_gradient=True,
            add_subtle_noise=True,
            debug_dir=debug_dir
        )
        
        # Save result
        cv2.imwrite("../outputs/debug_bg_v2/result.jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("\nResults saved to outputs/debug_bg_v2/")
        print("Metrics:", metrics)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from utils/ directory or adjust imports")
        sys.exit(1)

