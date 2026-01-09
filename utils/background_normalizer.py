"""
Background Normalization Module - Production Grade Person Segmentation

Replaces background with biometric light-gray (#F5F6F8) while preserving:
- Full person silhouette (hair, ears, neck, SHOULDERS, upper torso)
- Natural edge blending without oval/face-only masking
- No facial retouching

Uses MediaPipe ImageSegmenter (Tasks API) for proper person segmentation.
Falls back to enhanced GrabCut if MediaPipe unavailable.

CRITICAL: This module does NOT use face landmarks for matte shape.
Face landmarks are only used for crop centering, NOT alpha mask generation.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import os
import time

# Debug flag - set to True to save intermediate images
DEBUG_PIPELINE = os.getenv("DEBUG_BG_PIPELINE", "false").lower() == "true"
DEBUG_OUTPUT_DIR = "outputs/debug_bg"

# Biometric background color: #F5F6F8
BIOMETRIC_BG_COLOR = (245, 246, 248)  # RGB

# Soft alpha thresholds (prevents hard cutoff edges)
ALPHA_LOW_THRESHOLD = 0.15   # Below this = 0 (background)
ALPHA_HIGH_THRESHOLD = 0.75  # Above this = 1 (foreground)

# Model path
SELFIE_SEGMENTER_MODEL = "models/selfie_segmenter.tflite"

# Try to import MediaPipe Tasks API
MP_SEGMENTER_AVAILABLE = False
ImageSegmenter = None
ImageSegmenterOptions = None
BaseOptions = None
MPImage = None

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions as MPBaseOptions
    from mediapipe import Image as MPImageClass
    
    ImageSegmenter = vision.ImageSegmenter
    ImageSegmenterOptions = vision.ImageSegmenterOptions
    BaseOptions = MPBaseOptions
    MPImage = MPImageClass
    
    # Check if model exists
    if os.path.exists(SELFIE_SEGMENTER_MODEL):
        MP_SEGMENTER_AVAILABLE = True
        print(f"✅ MediaPipe ImageSegmenter available (model: {SELFIE_SEGMENTER_MODEL})")
    else:
        print(f"⚠️ Selfie segmenter model not found at {SELFIE_SEGMENTER_MODEL}")
except ImportError as e:
    print(f"⚠️ MediaPipe Tasks API not available: {e}")
except Exception as e:
    print(f"⚠️ MediaPipe setup error: {e}")


def _ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)


def _debug_save(img: np.ndarray, filename: str):
    """Save debug image with validation"""
    if not DEBUG_PIPELINE:
        return
    
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(DEBUG_OUTPUT_DIR, filename)
    
    # Validate input
    if img is None:
        print(f"  [DEBUG] WARNING: Cannot save {filename} - image is None")
        return
    
    # Ensure contiguous
    img_save = np.ascontiguousarray(img.copy())
    
    # Convert float to uint8 if needed
    if img_save.dtype in [np.float32, np.float64]:
        if img_save.max() <= 1.0:
            img_save = (img_save * 255).astype(np.uint8)
        else:
            img_save = np.clip(img_save, 0, 255).astype(np.uint8)
    
    # Ensure uint8
    if img_save.dtype != np.uint8:
        img_save = img_save.astype(np.uint8)
    
    success = cv2.imwrite(path, img_save)
    if success:
        print(f"  [DEBUG] Saved: {path} (shape={img_save.shape}, dtype={img_save.dtype})")
    else:
        print(f"  [DEBUG] FAILED to save: {path}")


def _print_mask_stats(mask: np.ndarray, name: str) -> float:
    """Print mask statistics and return coverage percentage"""
    if mask is None:
        print(f"  [{name}] ERROR: mask is None")
        return 0.0
    
    mask_float = mask.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float = mask_float / 255.0
    
    min_val = float(mask_float.min())
    max_val = float(mask_float.max())
    mean_val = float(mask_float.mean())
    coverage = float(np.sum(mask_float > 0.5) / mask_float.size * 100)
    
    print(f"  [{name}] min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}, coverage={coverage:.1f}%")
    return coverage


def _validate_array(arr: np.ndarray, name: str, expected_ndim: int = None, expected_dtype=None) -> bool:
    """Validate array properties"""
    if arr is None:
        print(f"  [VALIDATE] {name}: ERROR - array is None")
        return False
    
    if expected_ndim is not None and arr.ndim != expected_ndim:
        print(f"  [VALIDATE] {name}: ERROR - expected ndim={expected_ndim}, got {arr.ndim}")
        return False
    
    if expected_dtype is not None and arr.dtype != expected_dtype:
        print(f"  [VALIDATE] {name}: WARNING - expected dtype={expected_dtype}, got {arr.dtype}")
    
    if np.any(np.isnan(arr)):
        print(f"  [VALIDATE] {name}: ERROR - contains NaN values")
        return False
    
    if np.any(np.isinf(arr)):
        print(f"  [VALIDATE] {name}: ERROR - contains Inf values")
        return False
    
    if not arr.flags['C_CONTIGUOUS']:
        print(f"  [VALIDATE] {name}: WARNING - not C-contiguous")
    
    return True


def get_person_mask_mediapipe(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Get person segmentation mask using MediaPipe ImageSegmenter (Tasks API).
    
    Returns:
        Float32 mask (H, W) with values [0..1], or None if failed.
        1.0 = person, 0.0 = background
    """
    if not MP_SEGMENTER_AVAILABLE:
        print("  ⚠️ MediaPipe ImageSegmenter not available")
        return None
    
    try:
        # Create segmenter options
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=SELFIE_SEGMENTER_MODEL),
            output_category_mask=False,
            output_confidence_masks=True
        )
        
        # Create segmenter
        with ImageSegmenter.create_from_options(options) as segmenter:
            # Convert numpy array to MediaPipe Image
            mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=image_rgb)
            
            # Segment
            result = segmenter.segment(mp_image)
            
            # Get confidence mask (person probability)
            if result.confidence_masks and len(result.confidence_masks) > 0:
                mask_mp = result.confidence_masks[0]
                mask = np.array(mask_mp.numpy_view()).astype(np.float32)
                print(f"  ✅ MediaPipe segmentation successful: shape={mask.shape}")
                return mask
            else:
                print("  ⚠️ No confidence masks returned")
                return None
                
    except Exception as e:
        print(f"  ⚠️ MediaPipe segmentation failed: {e}")
        return None


def get_person_mask_grabcut(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Enhanced GrabCut fallback for person segmentation.
    
    CRITICAL: Uses FULL person initialization, not face-only.
    This ensures shoulders and torso are included.
    
    Returns:
        Float32 mask (H, W) with values [0..1], or None if failed.
    """
    h, w = image_rgb.shape[:2]
    
    try:
        # Convert to BGR for GrabCut
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Initialize mask with probable background
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:] = cv2.GC_PR_BGD  # Probable background
        
        # CRITICAL: Wide initialization to include FULL PERSON (head to shoulders)
        # For biometric photo: person is centered
        margin_x = int(w * 0.03)  # Only 3% margin on sides
        margin_top = int(h * 0.01)  # Very small top margin (keep all hair)
        margin_bottom = int(h * 0.01)  # Very small bottom margin (keep shoulders)
        
        rect = (margin_x, margin_top, w - 2 * margin_x, h - margin_top - margin_bottom)
        
        # Set most of the image as probable foreground (person fills most of biometric crop)
        center_x = w // 2
        
        # Large ellipse covering entire person (head + neck + shoulders)
        # This is the key to avoiding "floating head"
        cv2.ellipse(mask, (center_x, int(h * 0.45)), (int(w * 0.45), int(h * 0.45)), 
                    0, 0, 360, cv2.GC_PR_FGD, -1)
        
        # Definite foreground in face area (center-upper region)
        face_center_y = int(h * 0.30)
        cv2.ellipse(mask, (center_x, face_center_y), (int(w * 0.20), int(h * 0.15)), 
                    0, 0, 360, cv2.GC_FGD, -1)
        
        # Definite foreground in chest/shoulder area
        chest_center_y = int(h * 0.70)
        cv2.ellipse(mask, (center_x, chest_center_y), (int(w * 0.35), int(h * 0.20)), 
                    0, 0, 360, cv2.GC_FGD, -1)
        
        # Set corners as definite background (these are certainly not person)
        corner_h = int(h * 0.08)
        corner_w = int(w * 0.15)
        mask[0:corner_h, 0:corner_w] = cv2.GC_BGD
        mask[0:corner_h, w-corner_w:w] = cv2.GC_BGD
        
        # Initialize models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Run GrabCut with mask initialization (more iterations for better result)
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)
        
        # Create soft mask from GrabCut result
        # GC_FGD=1, GC_PR_FGD=3 are foreground
        mask_binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Convert to float [0..1]
        mask_float = mask_binary.astype(np.float32) / 255.0
        
        print(f"  ✅ GrabCut segmentation successful: shape={mask_float.shape}")
        return mask_float
        
    except Exception as e:
        print(f"  ⚠️ GrabCut segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_soft_alpha(mask: np.ndarray, t0: float = ALPHA_LOW_THRESHOLD, t1: float = ALPHA_HIGH_THRESHOLD) -> np.ndarray:
    """
    Create soft alpha from raw mask using threshold normalization.
    
    This prevents hard cutoff edges by creating a smooth transition:
    - Values below t0 become 0
    - Values above t1 become 1
    - Values between t0 and t1 are linearly interpolated
    
    Args:
        mask: Float32 mask [0..1]
        t0: Low threshold (below = 0)
        t1: High threshold (above = 1)
    
    Returns:
        Float32 soft alpha [0..1]
    """
    # Ensure float32
    mask_float = mask.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float = mask_float / 255.0
    
    # Apply soft threshold: alpha = clip((mask - t0) / (t1 - t0), 0, 1)
    alpha = np.clip((mask_float - t0) / (t1 - t0), 0.0, 1.0)
    
    return alpha.astype(np.float32)


def refine_alpha_mask(alpha: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Refine alpha mask with morphology and feathering.
    
    Steps:
    1. Morphological close (fill gaps in person)
    2. Morphological open (remove noise)
    3. Light dilation (recover edges, especially hair)
    4. Gaussian feather for natural edges
    
    IMPORTANT: No erosion that cuts person silhouette!
    
    Args:
        alpha: Float32 mask [0..1]
        sigma: Gaussian feather sigma (default 3.0)
    
    Returns:
        Refined float32 alpha [0..1]
    """
    h, w = alpha.shape[:2]
    
    # Ensure float32 [0..1]
    alpha_float = alpha.astype(np.float32)
    if alpha_float.max() > 1.0:
        alpha_float = alpha_float / 255.0
    
    # Convert to uint8 for morphological operations
    alpha_uint8 = (alpha_float * 255).astype(np.uint8)
    
    # Step 1: Morphological CLOSE to fill gaps (especially in hair/shoulders)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Step 2: Morphological OPEN to remove noise (small isolated pixels)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Step 3: Light dilation to recover edges (especially hair and shoulders)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha_uint8 = cv2.dilate(alpha_uint8, kernel_dilate, iterations=2)
    
    # Step 4: Bilateral filter (preserves edges better than Gaussian for initial smoothing)
    alpha_uint8 = cv2.bilateralFilter(alpha_uint8, 7, 75, 75)
    
    # Convert back to float
    alpha_refined = alpha_uint8.astype(np.float32) / 255.0
    
    # Step 5: Gaussian feather for natural edges
    alpha_refined = cv2.GaussianBlur(alpha_refined, (0, 0), sigmaX=sigma)
    
    # Final clamp
    alpha_refined = np.clip(alpha_refined, 0.0, 1.0).astype(np.float32)
    
    return alpha_refined


def composite_with_background(
    foreground_rgb: np.ndarray,
    alpha: np.ndarray,
    bg_color: Tuple[int, int, int] = BIOMETRIC_BG_COLOR
) -> np.ndarray:
    """
    Composite foreground with solid background color.
    
    CRITICAL: All operations in float32 to prevent overflow/corruption.
    
    Args:
        foreground_rgb: RGB image uint8 (H, W, 3)
        alpha: Float32 mask (H, W) with values [0..1]
        bg_color: RGB tuple (R, G, B) 0-255
    
    Returns:
        RGB image uint8 (H, W, 3)
    """
    h, w = foreground_rgb.shape[:2]
    
    # ========================================================================
    # STEP 1: Validate inputs
    # ========================================================================
    assert foreground_rgb.ndim == 3 and foreground_rgb.shape[2] == 3, f"FG shape: {foreground_rgb.shape}"
    assert alpha.ndim == 2, f"Alpha ndim: {alpha.ndim}"
    assert alpha.shape == (h, w), f"Alpha shape mismatch: {alpha.shape} vs ({h}, {w})"
    
    # ========================================================================
    # STEP 2: Convert foreground to float32 [0..1]
    # ========================================================================
    fg = foreground_rgb.astype(np.float32) / 255.0
    
    # ========================================================================
    # STEP 3: Create background as float32 [0..1] - EXPLICIT channel assignment
    # ========================================================================
    # CRITICAL: Do NOT use np.full_like with tuple - it causes broadcasting issues
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = bg_color[0] / 255.0  # R
    bg[:, :, 1] = bg_color[1] / 255.0  # G
    bg[:, :, 2] = bg_color[2] / 255.0  # B
    
    # ========================================================================
    # STEP 4: Ensure alpha is float32 [0..1] and expand dims
    # ========================================================================
    alpha_float = alpha.astype(np.float32)
    if alpha_float.max() > 1.0:
        alpha_float = alpha_float / 255.0
    alpha_float = np.clip(alpha_float, 0.0, 1.0)
    
    # Expand to (H, W, 1) for broadcasting
    alpha_3d = alpha_float[:, :, np.newaxis]
    
    # ========================================================================
    # STEP 5: Composite in float space
    # ========================================================================
    out = fg * alpha_3d + bg * (1.0 - alpha_3d)
    
    # ========================================================================
    # STEP 6: Check for NaN/Inf
    # ========================================================================
    if np.any(np.isnan(out)):
        raise RuntimeError(f"Composite contains {np.sum(np.isnan(out))} NaN values")
    if np.any(np.isinf(out)):
        raise RuntimeError(f"Composite contains {np.sum(np.isinf(out))} Inf values")
    
    # ========================================================================
    # STEP 7: Convert back to uint8
    # ========================================================================
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    
    # ========================================================================
    # STEP 8: Ensure contiguous
    # ========================================================================
    out = np.ascontiguousarray(out)
    
    return out


def replace_background_biometric(
    image_bgr: np.ndarray,
    bg_color: Tuple[int, int, int] = BIOMETRIC_BG_COLOR,
    job_id: str = "test"
) -> Tuple[np.ndarray, Dict]:
    """
    Replace background with biometric-compliant solid color.
    
    Uses person segmentation to preserve full silhouette:
    - Hair, ears, neck
    - SHOULDERS and upper torso (critical!)
    
    No facial retouching is applied.
    
    Args:
        image_bgr: Input BGR image (from cv2.imread)
        bg_color: Background RGB color (default: #F5F6F8)
        job_id: Job ID for debug file naming
    
    Returns:
        (output_bgr, metrics_dict)
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"[BG_REPLACE] Starting biometric background replacement")
    print(f"  Job ID: {job_id}")
    print(f"  Target BG: #{bg_color[0]:02X}{bg_color[1]:02X}{bg_color[2]:02X}")
    print(f"{'='*60}")
    
    # ========================================================================
    # STEP 0: Validate input
    # ========================================================================
    if image_bgr is None:
        raise ValueError("Input image is None")
    
    if not _validate_array(image_bgr, "input_image", expected_ndim=3):
        raise ValueError("Input image validation failed")
    
    if image_bgr.shape[2] != 3:
        raise ValueError(f"Input must be 3-channel, got {image_bgr.shape[2]} channels")
    
    if image_bgr.dtype != np.uint8:
        print(f"  ⚠️ Converting input from {image_bgr.dtype} to uint8")
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    
    h, w = image_bgr.shape[:2]
    print(f"\n[INPUT] Size: {w}x{h}, dtype: {image_bgr.dtype}")
    
    # Make contiguous copy
    image_bgr = np.ascontiguousarray(image_bgr.copy())
    
    # Debug: Save input
    _debug_save(image_bgr, f"{job_id}_01_input_bgr.png")
    
    # Convert BGR to RGB for processing
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    
    # ========================================================================
    # STEP 1: Get person segmentation mask
    # ========================================================================
    print("\n[STEP 1] Person segmentation...")
    
    raw_mask = get_person_mask_mediapipe(image_rgb)
    seg_model = "mediapipe_selfie"
    
    # Fallback to GrabCut if MediaPipe fails
    if raw_mask is None:
        print("  Trying GrabCut fallback...")
        raw_mask = get_person_mask_grabcut(image_rgb)
        seg_model = "grabcut_fallback"
    
    if raw_mask is None:
        print("  ⚠️ All segmentation methods failed, returning original")
        return image_bgr, {
            "success": False,
            "error": "segmentation_failed",
            "seg_model": "none"
        }
    
    # Validate raw mask
    if not _validate_array(raw_mask, "raw_mask", expected_ndim=2):
        return image_bgr, {"success": False, "error": "invalid_raw_mask", "seg_model": seg_model}
    
    coverage_raw = _print_mask_stats(raw_mask, "RAW MASK")
    _debug_save((raw_mask * 255).astype(np.uint8), f"{job_id}_02_mask_raw.png")
    
    # ========================================================================
    # STEP 2: Validate mask coverage
    # ========================================================================
    print("\n[STEP 2] Validating coverage...")
    
    if coverage_raw < 5.0:
        print(f"  ⚠️ Coverage too low ({coverage_raw:.1f}% < 5%)")
        return image_bgr, {
            "success": False,
            "error": "mask_coverage_too_low",
            "coverage": coverage_raw,
            "seg_model": seg_model
        }
    
    if coverage_raw > 95.0:
        print(f"  ⚠️ Coverage too high ({coverage_raw:.1f}% > 95%)")
        return image_bgr, {
            "success": False,
            "error": "mask_coverage_too_high",
            "coverage": coverage_raw,
            "seg_model": seg_model
        }
    
    print(f"  ✅ Coverage OK: {coverage_raw:.1f}%")
    
    # ========================================================================
    # STEP 3: Create soft alpha (smooth transitions)
    # ========================================================================
    print("\n[STEP 3] Creating soft alpha...")
    
    soft_alpha = create_soft_alpha(raw_mask, t0=ALPHA_LOW_THRESHOLD, t1=ALPHA_HIGH_THRESHOLD)
    coverage_soft = _print_mask_stats(soft_alpha, "SOFT ALPHA")
    _debug_save((soft_alpha * 255).astype(np.uint8), f"{job_id}_03_alpha_soft.png")
    
    # ========================================================================
    # STEP 4: Refine alpha (morphology + feather)
    # ========================================================================
    print("\n[STEP 4] Refining alpha...")
    
    refined_alpha = refine_alpha_mask(soft_alpha, sigma=3.5)
    coverage_refined = _print_mask_stats(refined_alpha, "REFINED ALPHA")
    _debug_save((refined_alpha * 255).astype(np.uint8), f"{job_id}_04_alpha_refined.png")
    
    # ========================================================================
    # STEP 5: Composite with background
    # ========================================================================
    print("\n[STEP 5] Compositing...")
    print(f"  FG: {image_rgb.shape}, {image_rgb.dtype}")
    print(f"  Alpha: {refined_alpha.shape}, {refined_alpha.dtype}")
    print(f"  BG color: RGB{bg_color}")
    
    output_rgb = composite_with_background(image_rgb, refined_alpha, bg_color)
    
    if not _validate_array(output_rgb, "output_rgb", expected_ndim=3):
        raise RuntimeError("Composite output validation failed")
    
    _debug_save(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR), f"{job_id}_05_composited.png")
    
    # ========================================================================
    # STEP 6: Convert back to BGR
    # ========================================================================
    print("\n[STEP 6] Final conversion...")
    
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    output_bgr = np.ascontiguousarray(output_bgr)
    
    # Final validation
    assert output_bgr.shape == (h, w, 3), f"Output shape: {output_bgr.shape}"
    assert output_bgr.dtype == np.uint8, f"Output dtype: {output_bgr.dtype}"
    assert output_bgr.flags['C_CONTIGUOUS'], "Output not contiguous"
    
    _debug_save(output_bgr, f"{job_id}_06_final_bgr.png")
    
    # ========================================================================
    # Metrics
    # ========================================================================
    elapsed = time.time() - start_time
    
    metrics = {
        "success": True,
        "seg_model": seg_model,
        "mask_coverage_raw": round(float(coverage_raw), 1),
        "mask_coverage_soft": round(float(coverage_soft), 1),
        "mask_coverage_refined": round(float(coverage_refined), 1),
        "alpha_thresholds": {"low": ALPHA_LOW_THRESHOLD, "high": ALPHA_HIGH_THRESHOLD},
        "bg_color_rgb": bg_color,
        "bg_color_hex": f"#{bg_color[0]:02X}{bg_color[1]:02X}{bg_color[2]:02X}",
        "processing_time_ms": round(elapsed * 1000, 1)
    }
    
    print(f"\n{'='*60}")
    print(f"[BG_REPLACE] Complete in {elapsed*1000:.0f}ms")
    print(f"  Model: {seg_model}")
    print(f"  Coverage: raw={coverage_raw:.1f}% → refined={coverage_refined:.1f}%")
    print(f"  Output: {output_bgr.shape}, {output_bgr.dtype}, contiguous={output_bgr.flags['C_CONTIGUOUS']}")
    print(f"{'='*60}\n")
    
    return output_bgr, metrics


# ============================================================================
# Legacy compatibility wrapper
# ============================================================================

def normalize_background(
    input_img: np.ndarray,
    landmarks_or_bbox: Optional[Dict] = None,
    *,
    bg_color: Tuple[int, int, int] = BIOMETRIC_BG_COLOR,
    feather_px: int = 8,
    job_id: str = "test"
) -> Tuple[np.ndarray, Dict]:
    """
    Legacy wrapper for background normalization.
    
    NOTE: landmarks_or_bbox is IGNORED - we use full person segmentation,
    NOT face-only masks. This prevents "floating head" artifacts.
    """
    # Call the main function (landmarks are not used for matte shape)
    output_bgr, metrics = replace_background_biometric(
        input_img,
        bg_color=bg_color,
        job_id=job_id
    )
    
    # Add legacy metric fields for compatibility
    if "seg_model" in metrics:
        metrics["seg_model_used"] = metrics["seg_model"]
    if "mask_coverage_refined" in metrics:
        metrics["alpha_mean"] = metrics["mask_coverage_refined"] / 100.0
    metrics["alpha_edge_softness"] = 0.5  # Placeholder
    metrics["seg_reliability"] = 0.8 if metrics.get("success", False) else 0.0
    
    return output_bgr, metrics


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    import sys
    import glob
    
    # Enable debug
    os.environ["DEBUG_BG_PIPELINE"] = "true"
    DEBUG_PIPELINE = True
    
    print("="*60)
    print("Background Normalizer - Standalone Test")
    print("="*60)
    
    # Find test image
    test_images = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.jpeg") + glob.glob("uploads/*.png")
    
    if not test_images:
        print("No test images found in uploads/")
        sys.exit(1)
    
    test_path = test_images[0]
    print(f"\nTesting with: {test_path}")
    
    # Load image
    img_bgr = cv2.imread(test_path)
    if img_bgr is None:
        print(f"Failed to load {test_path}")
        sys.exit(1)
    
    print(f"Loaded: {img_bgr.shape}, {img_bgr.dtype}")
    
    # Run background replacement
    output_bgr, metrics = replace_background_biometric(img_bgr, job_id="standalone_test")
    
    # Save results
    os.makedirs("outputs/debug_bg", exist_ok=True)
    
    cv2.imwrite("outputs/debug_bg/standalone_result.jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite("outputs/debug_bg/standalone_result.png", output_bgr)
    
    print("\nResults saved:")
    print("  - outputs/debug_bg/standalone_result.jpg")
    print("  - outputs/debug_bg/standalone_result.png")
    
    # Verify
    print("\nMetrics:", metrics)
    
    # Check corners
    verify = cv2.imread("outputs/debug_bg/standalone_result.png")
    if verify is not None:
        print("\nCorner colors (BGR) - should be ~(248, 246, 245) for #F5F6F8:")
        corners = [
            ("TL", verify[5:15, 5:15]),
            ("TR", verify[5:15, -15:-5]),
            ("BL", verify[-15:-5, 5:15]),
            ("BR", verify[-15:-5, -15:-5])
        ]
        for name, c in corners:
            mean_color = c.mean(axis=(0,1))
            print(f"  {name}: {mean_color}")
