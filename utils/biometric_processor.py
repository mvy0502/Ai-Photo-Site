"""
Biometric Photo Processor

Complete pipeline for producing biometric-ready photos:
1. Face detection with MediaPipe FaceLandmarker
2. Wide crop computation (preserves hair + shoulders)
3. Background removal via PhotoRoom API
4. Mask cleanup (connected component filtering)
5. Compositing with biometric background
6. Canvas padding and centering

Environment Variables:
    DEBUG_BG: Set to "1" to save debug images
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import time

from .photoroom_client import remove_background_transparent

# =============================================================================
# Configuration
# =============================================================================

# Biometric background color (RGB)
BIOMETRIC_BG_COLOR_RGB = (245, 246, 248)  # #F5F6F8
BIOMETRIC_BG_COLOR_BGR = (248, 246, 245)  # BGR for OpenCV

# =============================================================================
# PORTRAIT CROP MARGINS (relative to face bounding box)
# =============================================================================
# CRITICAL: These margins define the crop sent to PhotoRoom.
# The crop MUST include head + shoulders + upper torso.
# If margins are too tight, PhotoRoom output will have oversized head
# and cut shoulders - looking unprofessional.
#
# DO NOT reduce these margins without understanding the impact.
# =============================================================================
PORTRAIT_CROP_MARGIN_TOP = 0.60      # 60% of face height above (for hair + forehead space)
PORTRAIT_CROP_MARGIN_LEFT = 0.55     # 55% of face width on left (for ears + some space)
PORTRAIT_CROP_MARGIN_RIGHT = 0.55    # 55% of face width on right (for ears + some space)
PORTRAIT_CROP_MARGIN_BOTTOM = 1.10   # 110% of face height below (CRITICAL for shoulders)

# Target canvas size
TARGET_CANVAS_WIDTH = 600
TARGET_CANVAS_HEIGHT = 600

# Mask cleanup parameters
MORPH_CLOSE_KERNEL_SIZE = 7
MORPH_CLOSE_ITERATIONS = 2
FEATHER_SIGMA = 1.5
ERODE_PIXELS = 1  # Slight erosion to reduce halo

# Minimum connected component size (as fraction of image area)
MIN_COMPONENT_FRACTION = 0.05

# Debug
DEBUG_ENABLED = os.getenv("DEBUG_BG", "0") == "1"
DEBUG_OUTPUT_DIR = "outputs/debug_biometric"


def _debug_save(image: np.ndarray, filename: str, job_id: str = ""):
    """Save debug image if debug mode is enabled"""
    if not DEBUG_ENABLED:
        return
    
    output_dir = os.path.join(DEBUG_OUTPUT_DIR, job_id) if job_id else DEBUG_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    # Handle different image types
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image_save = (image * 255).astype(np.uint8)
        else:
            image_save = image.astype(np.uint8)
    else:
        image_save = image
    
    cv2.imwrite(filepath, image_save)
    print(f"  [DEBUG] Saved: {filepath}")


# =============================================================================
# Face Detection and Crop Computation
# =============================================================================

def detect_face_landmarks(image_bgr: np.ndarray) -> Optional[Dict]:
    """
    Detect face landmarks using MediaPipe FaceLandmarker.
    
    Returns:
        Dict with 'bbox', 'center', 'raw_landmarks' or None if no face
    """
    try:
        from .analyzer_v2 import get_analyzer
        
        analyzer = get_analyzer()
        if not analyzer or not analyzer.initialized:
            print("  ⚠️ FaceLandmarker not available")
            return None
        
        result = analyzer.analyze(image_bgr)
        
        if not result or not result.get("face_detected"):
            return None
        
        face_bbox = result.get("face_bbox")
        raw_landmarks = result.get("raw_landmarks")
        
        if not face_bbox:
            return None
        
        # Face center from bbox
        height, width = image_bgr.shape[:2]
        center_x = face_bbox[0] + face_bbox[2] / 2  # xmin + width/2
        center_y = face_bbox[1] + face_bbox[3] / 2  # ymin + height/2
        
        return {
            "bbox": face_bbox,  # (xmin, ymin, width, height) normalized
            "center": (center_x, center_y),  # normalized
            "center_pixel": (int(center_x * width), int(center_y * height)),
            "raw_landmarks": raw_landmarks,
            "image_size": (width, height)
        }
    
    except Exception as e:
        print(f"  ⚠️ Face detection error: {e}")
        return None


def compute_portrait_crop_box(
    face_bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
    """
    Compute PORTRAIT crop box for PhotoRoom input.
    
    CRITICAL: This crop MUST include head + shoulders + upper torso.
    PhotoRoom needs a "portrait-style" image, NOT a tight passport crop.
    
    The final biometric framing happens AFTER PhotoRoom processing,
    not before. If we send a tight crop to PhotoRoom:
    - Head appears oversized
    - Shoulders get cut
    - Output looks unprofessional
    
    Margins are relative to face bounding box dimensions:
    - top: 60% of face height (hair + forehead space)
    - sides: 55% of face width each (ears + space)
    - bottom: 110% of face height (SHOULDERS - most important)
    
    Args:
        face_bbox: (xmin, ymin, width, height) in normalized coordinates [0..1]
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        (x, y, w, h) in pixel coordinates, clamped to image bounds
    """
    # Convert normalized bbox to pixels
    face_x = int(face_bbox[0] * image_width)
    face_y = int(face_bbox[1] * image_height)
    face_w = int(face_bbox[2] * image_width)
    face_h = int(face_bbox[3] * image_height)
    
    # Compute margins in pixels using PORTRAIT crop margins
    # These are intentionally generous to include shoulders
    top_margin = int(face_h * PORTRAIT_CROP_MARGIN_TOP)
    left_margin = int(face_w * PORTRAIT_CROP_MARGIN_LEFT)
    right_margin = int(face_w * PORTRAIT_CROP_MARGIN_RIGHT)
    bottom_margin = int(face_h * PORTRAIT_CROP_MARGIN_BOTTOM)  # Large for shoulders!
    
    # Compute crop box
    crop_x = max(0, face_x - left_margin)
    crop_y = max(0, face_y - top_margin)
    crop_x2 = min(image_width, face_x + face_w + right_margin)
    crop_y2 = min(image_height, face_y + face_h + bottom_margin)
    
    crop_w = crop_x2 - crop_x
    crop_h = crop_y2 - crop_y
    
    return (crop_x, crop_y, crop_w, crop_h)


def apply_wide_crop(
    image_bgr: np.ndarray,
    crop_box: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, Dict]:
    """
    Apply wide crop to image.
    
    Returns:
        (cropped_image, crop_info_dict)
    """
    x, y, w, h = crop_box
    cropped = image_bgr[y:y+h, x:x+w].copy()
    
    crop_info = {
        "crop_x": x,
        "crop_y": y,
        "crop_width": w,
        "crop_height": h,
        "original_width": image_bgr.shape[1],
        "original_height": image_bgr.shape[0]
    }
    
    return cropped, crop_info


# =============================================================================
# Mask Processing
# =============================================================================

def extract_alpha_from_rgba(image_rgba: np.ndarray) -> np.ndarray:
    """Extract alpha channel from RGBA image as float [0..1]"""
    if image_rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got {image_rgba.shape[2]} channels")
    
    alpha = image_rgba[:, :, 3].astype(np.float32) / 255.0
    return alpha


def filter_connected_components(
    alpha_mask: np.ndarray,
    face_center_pixel: Tuple[int, int],
    min_component_fraction: float = MIN_COMPONENT_FRACTION
) -> np.ndarray:
    """
    Keep only the connected component containing the face center.
    Removes disconnected blobs (shadows, artifacts).
    
    Args:
        alpha_mask: Float mask [0..1]
        face_center_pixel: (x, y) face center in mask coordinates
        min_component_fraction: Minimum size to consider
    
    Returns:
        Cleaned alpha mask [0..1]
    """
    height, width = alpha_mask.shape[:2]
    face_x, face_y = face_center_pixel
    
    # Clamp face center to valid range
    face_x = max(0, min(face_x, width - 1))
    face_y = max(0, min(face_y, height - 1))
    
    # Binarize mask
    binary_mask = (alpha_mask > 0.5).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    if num_labels <= 1:
        # Only background or single component
        return alpha_mask
    
    # Find component containing face center
    face_label = labels[face_y, face_x]
    
    if face_label == 0:
        # Face center is on background - find nearest/largest component
        # This can happen if face center is slightly misaligned
        print(f"  ⚠️ Face center ({face_x}, {face_y}) on background, finding largest component")
        
        # Find largest non-background component
        largest_label = 0
        largest_area = 0
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = label_id
        
        face_label = largest_label
    
    if face_label == 0:
        # No valid components found
        return alpha_mask
    
    # Create mask with only the face component
    component_mask = (labels == face_label).astype(np.float32)
    
    # Apply to original alpha (preserves soft edges)
    cleaned_alpha = alpha_mask * component_mask
    
    if DEBUG_ENABLED:
        removed_pixels = np.sum((alpha_mask > 0.5) & (component_mask == 0))
        print(f"  [CC Filter] Kept component {face_label}, removed {removed_pixels} pixels from other components")
    
    return cleaned_alpha


def refine_alpha_mask(
    alpha: np.ndarray,
    close_kernel_size: int = MORPH_CLOSE_KERNEL_SIZE,
    close_iterations: int = MORPH_CLOSE_ITERATIONS,
    feather_sigma: float = FEATHER_SIGMA,
    erode_pixels: int = ERODE_PIXELS
) -> np.ndarray:
    """
    Refine alpha mask: close holes, feather edges, reduce halo.
    
    Args:
        alpha: Float mask [0..1]
        close_kernel_size: Morphological close kernel size
        close_iterations: Number of close iterations
        feather_sigma: Gaussian blur sigma for edge feathering
        erode_pixels: Pixels to erode (reduces halo)
    
    Returns:
        Refined alpha mask [0..1]
    """
    # Convert to uint8 for morphology
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    
    # Morphological close to fill small holes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (close_kernel_size, close_kernel_size)
    )
    alpha_uint8 = cv2.morphologyEx(
        alpha_uint8, cv2.MORPH_CLOSE, kernel, 
        iterations=close_iterations
    )
    
    # Slight erosion to reduce halo
    if erode_pixels > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_pixels * 2 + 1, erode_pixels * 2 + 1)
        )
        alpha_uint8 = cv2.erode(alpha_uint8, erode_kernel, iterations=1)
    
    # Convert back to float
    alpha_refined = alpha_uint8.astype(np.float32) / 255.0
    
    # Feather edges
    if feather_sigma > 0:
        alpha_refined = cv2.GaussianBlur(
            alpha_refined, (0, 0), sigmaX=feather_sigma
        )
    
    # Final clamp
    alpha_refined = np.clip(alpha_refined, 0.0, 1.0)
    
    return alpha_refined


# =============================================================================
# Compositing
# =============================================================================

def composite_with_background(
    foreground_rgb: np.ndarray,
    alpha: np.ndarray,
    background_color_rgb: Tuple[int, int, int] = BIOMETRIC_BG_COLOR_RGB
) -> np.ndarray:
    """
    Composite foreground onto solid background color.
    
    Args:
        foreground_rgb: RGB image (H, W, 3) uint8
        alpha: Alpha mask (H, W) float [0..1]
        background_color_rgb: Background RGB tuple
    
    Returns:
        Composited RGB image uint8
    """
    height, width = foreground_rgb.shape[:2]
    
    # Convert to float
    fg = foreground_rgb.astype(np.float32) / 255.0
    
    # Create background
    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:, :, 0] = background_color_rgb[0] / 255.0
    bg[:, :, 1] = background_color_rgb[1] / 255.0
    bg[:, :, 2] = background_color_rgb[2] / 255.0
    
    # Expand alpha for broadcasting
    alpha_3d = alpha[:, :, np.newaxis]
    
    # Composite
    composited = fg * alpha_3d + bg * (1.0 - alpha_3d)
    
    # Convert back to uint8
    composited = np.clip(composited * 255, 0, 255).astype(np.uint8)
    
    return composited


# =============================================================================
# Canvas Placement
# =============================================================================

def place_on_biometric_canvas(
    image_rgb: np.ndarray,
    face_center_in_image: Tuple[int, int],
    face_height_in_image: int,
    canvas_width: int = TARGET_CANVAS_WIDTH,
    canvas_height: int = TARGET_CANVAS_HEIGHT,
    background_color_rgb: Tuple[int, int, int] = BIOMETRIC_BG_COLOR_RGB
) -> np.ndarray:
    """
    Place PhotoRoom output onto biometric canvas WITHOUT aggressive cropping.
    
    CRITICAL RULES:
    1. DO NOT crop the image - only scale and position
    2. Scale so image width fills ~80-85% of canvas
    3. Center horizontally
    4. Position vertically so eyes are at ~45-50% of canvas height
    5. Shoulders MUST remain visible
    
    This function assumes input already has background removed and
    includes head + shoulders from the portrait crop.
    
    Args:
        image_rgb: PhotoRoom output (RGB, background already replaced)
        face_center_in_image: (x, y) face center in source image
        face_height_in_image: Face height in pixels (for eye position estimation)
        canvas_width: Target canvas width
        canvas_height: Target canvas height
        background_color_rgb: Canvas background color
    
    Returns:
        Canvas with image placed (RGB uint8)
    """
    src_height, src_width = image_rgb.shape[:2]
    face_x, face_y = face_center_in_image
    
    # ==========================================================================
    # SCALING STRATEGY
    # Scale image so width fills 80-85% of canvas width
    # This ensures shoulders are visible and head is not oversized
    # ==========================================================================
    target_image_width = int(canvas_width * 0.82)  # 82% of canvas width
    scale = target_image_width / src_width
    
    # Limit maximum scale to avoid upscaling too much
    scale = min(scale, 1.2)
    
    # Also ensure image fits vertically (with some margin)
    max_height = int(canvas_height * 0.95)
    if src_height * scale > max_height:
        scale = max_height / src_height
    
    # Scaled dimensions
    scaled_width = int(src_width * scale)
    scaled_height = int(src_height * scale)
    
    # Resize image
    scaled_image = cv2.resize(
        image_rgb, 
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LANCZOS4
    )
    
    # ==========================================================================
    # POSITIONING STRATEGY
    # - Center horizontally
    # - Position eyes at approximately 45-50% of canvas height
    # - Eye position is estimated as face_center_y - 0.1 * face_height
    # ==========================================================================
    
    # Calculate face/eye position in scaled image
    scaled_face_y = int(face_y * scale)
    scaled_face_height = int(face_height_in_image * scale)
    
    # Estimate eye position (slightly above face center)
    estimated_eye_y = scaled_face_y - int(scaled_face_height * 0.10)
    
    # Target eye position: 47% from top of canvas
    target_eye_y = int(canvas_height * 0.47)
    
    # Horizontal centering
    offset_x = (canvas_width - scaled_width) // 2
    
    # Vertical positioning based on eye line
    offset_y = target_eye_y - estimated_eye_y
    
    # Adjust if image would go off canvas bottom (prioritize showing shoulders)
    if offset_y + scaled_height > canvas_height:
        # Shift up, but not so much that we cut the top
        offset_y = canvas_height - scaled_height
        offset_y = max(0, offset_y)  # Don't go negative
    
    # ==========================================================================
    # CREATE CANVAS AND PLACE IMAGE
    # No cropping - just placement
    # ==========================================================================
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:, :, 0] = background_color_rgb[0]
    canvas[:, :, 1] = background_color_rgb[1]
    canvas[:, :, 2] = background_color_rgb[2]
    
    # Calculate valid regions (handle edge cases)
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(scaled_width, canvas_width - offset_x)
    src_y2 = min(scaled_height, canvas_height - offset_y)
    
    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    # Paste image onto canvas (no cropping, just placement)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]
    
    if DEBUG_ENABLED:
        print(f"  [Canvas] Scale: {scale:.2f}, Offset: ({offset_x}, {offset_y})")
        print(f"  [Canvas] Eye target: {target_eye_y}px, Actual: {offset_y + estimated_eye_y}px")
    
    return canvas


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_biometric_photo(
    image_bgr: np.ndarray,
    job_id: str = "",
    canvas_width: int = TARGET_CANVAS_WIDTH,
    canvas_height: int = TARGET_CANVAS_HEIGHT,
    background_color_rgb: Tuple[int, int, int] = BIOMETRIC_BG_COLOR_RGB
) -> Tuple[np.ndarray, Dict]:
    """
    Complete biometric photo processing pipeline.
    
    ==========================================================================
    PIPELINE OVERVIEW (DO NOT MODIFY WITHOUT UNDERSTANDING)
    ==========================================================================
    
    STEP 1: Face Detection
        - Detect face using MediaPipe FaceLandmarker
        - Get face bounding box and center
    
    STEP 2: Portrait Crop (BEFORE PhotoRoom)
        - Compute GENEROUS crop that includes:
          * Hair (60% of face height above)
          * Sides (55% of face width each side)
          * SHOULDERS (110% of face height below) <-- CRITICAL
        - This is NOT the final biometric crop!
        - PhotoRoom needs portrait-style input, not passport crop
    
    STEP 3: PhotoRoom API
        - Send portrait crop to PhotoRoom (crop=false, size=full)
        - Do NOT resize before sending
        - Get transparent PNG back
    
    STEP 4: Alpha Processing
        - Extract alpha from RGBA
        - Clean up disconnected components
        - Light refinement (no aggressive erosion)
    
    STEP 5: Compositing
        - Composite onto biometric gray background (#F5F6F8)
    
    STEP 6: Canvas Placement (AFTER PhotoRoom)
        - Place on target canvas (e.g., 600x600)
        - Scale to fill ~80-85% of canvas width
        - Position eyes at ~47% of canvas height
        - DO NOT crop - only scale and position
        - Shoulders MUST remain visible
    
    ==========================================================================
    
    Args:
        image_bgr: Input BGR image
        job_id: Job ID for debug file naming
        canvas_width: Target output width
        canvas_height: Target output height
        background_color_rgb: Background color RGB tuple
    
    Returns:
        (output_bgr, metrics_dict)
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"[BiometricProcessor] Starting pipeline")
    print(f"  Job ID: {job_id}")
    print(f"  Input: {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    print(f"  Target canvas: {canvas_width}x{canvas_height}")
    print(f"{'='*60}")
    
    _debug_save(image_bgr, "01_input.png", job_id)
    
    # =========================================================================
    # Step 1: Detect face
    # =========================================================================
    print("\n[Step 1] Detecting face landmarks...")
    
    face_info = detect_face_landmarks(image_bgr)
    
    if not face_info:
        print("  ⚠️ No face detected, returning original")
        return image_bgr, {
            "success": False,
            "error": "no_face_detected",
            "processing_time_ms": round((time.time() - start_time) * 1000, 1)
        }
    
    print(f"  Face bbox: {face_info['bbox']}")
    print(f"  Face center: {face_info['center_pixel']}")
    
    # =========================================================================
    # Step 2: Compute PORTRAIT crop (for PhotoRoom input)
    # =========================================================================
    # CRITICAL: This is NOT the final biometric crop!
    # We need a generous crop that includes SHOULDERS for PhotoRoom.
    # Final biometric framing happens in Step 8 (canvas placement).
    print("\n[Step 2] Computing PORTRAIT crop box (includes shoulders)...")
    
    image_height, image_width = image_bgr.shape[:2]
    crop_box = compute_portrait_crop_box(
        face_info["bbox"],
        image_width,
        image_height
    )
    
    # Calculate face dimensions for later use
    face_width_pixels = int(face_info["bbox"][2] * image_width)
    face_height_pixels = int(face_info["bbox"][3] * image_height)
    
    print(f"  Face size: {face_width_pixels}x{face_height_pixels} pixels")
    print(f"  Portrait crop: x={crop_box[0]}, y={crop_box[1]}, w={crop_box[2]}, h={crop_box[3]}")
    
    # =========================================================================
    # Step 3: Apply portrait crop
    # =========================================================================
    # This crop will be sent to PhotoRoom. It includes head + shoulders.
    print("\n[Step 3] Applying portrait crop...")
    
    cropped_image, crop_info = apply_wide_crop(image_bgr, crop_box)
    print(f"  Portrait crop size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    
    _debug_save(cropped_image, "02_portrait_crop.png", job_id)
    
    # Update face center to cropped coordinates
    face_center_in_crop = (
        face_info["center_pixel"][0] - crop_box[0],
        face_info["center_pixel"][1] - crop_box[1]
    )
    
    # Face height in cropped image (same as original, just different coordinate system)
    face_height_in_crop = face_height_pixels
    
    print(f"  Face center in crop: {face_center_in_crop}")
    print(f"  Face height in crop: {face_height_in_crop}px")
    
    # =========================================================================
    # Step 4: Send to PhotoRoom
    # =========================================================================
    print("\n[Step 4] Sending to PhotoRoom API...")
    
    # Encode cropped image to bytes
    _, encoded_bytes = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    image_bytes = encoded_bytes.tobytes()
    
    try:
        # Get transparent PNG from PhotoRoom (we composite ourselves)
        result_bytes, photoroom_metadata = remove_background_transparent(
            image_bytes,
            size="full"
        )
        print(f"  PhotoRoom response: {len(result_bytes)} bytes")
    except Exception as e:
        print(f"  ⚠️ PhotoRoom API failed: {e}")
        return image_bgr, {
            "success": False,
            "error": f"photoroom_api_failed: {str(e)}",
            "processing_time_ms": round((time.time() - start_time) * 1000, 1)
        }
    
    # =========================================================================
    # Step 5: Decode and extract alpha
    # =========================================================================
    print("\n[Step 5] Processing PhotoRoom result...")
    
    # Decode PNG
    nparr = np.frombuffer(result_bytes, np.uint8)
    result_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if result_rgba is None or result_rgba.shape[2] != 4:
        print(f"  ⚠️ Invalid PhotoRoom response (not RGBA)")
        return image_bgr, {
            "success": False,
            "error": "invalid_photoroom_response",
            "processing_time_ms": round((time.time() - start_time) * 1000, 1)
        }
    
    print(f"  Result RGBA: {result_rgba.shape}")
    
    # Extract RGB and alpha
    result_bgr = result_rgba[:, :, :3]
    raw_alpha = extract_alpha_from_rgba(result_rgba)
    
    _debug_save((raw_alpha * 255).astype(np.uint8), "03_raw_alpha.png", job_id)
    
    # =========================================================================
    # Step 6: Clean alpha mask
    # =========================================================================
    print("\n[Step 6] Cleaning alpha mask...")
    
    # Connected component filtering
    cleaned_alpha = filter_connected_components(
        raw_alpha,
        face_center_in_crop,
        min_component_fraction=MIN_COMPONENT_FRACTION
    )
    
    _debug_save((cleaned_alpha * 255).astype(np.uint8), "04_cleaned_alpha.png", job_id)
    
    # Refine edges
    refined_alpha = refine_alpha_mask(
        cleaned_alpha,
        close_kernel_size=MORPH_CLOSE_KERNEL_SIZE,
        close_iterations=MORPH_CLOSE_ITERATIONS,
        feather_sigma=FEATHER_SIGMA,
        erode_pixels=ERODE_PIXELS
    )
    
    alpha_coverage = float(np.mean(refined_alpha) * 100)
    print(f"  Alpha coverage: {alpha_coverage:.1f}%")
    
    _debug_save((refined_alpha * 255).astype(np.uint8), "05_refined_alpha.png", job_id)
    
    # =========================================================================
    # Step 7: Composite with background
    # =========================================================================
    print("\n[Step 7] Compositing with background...")
    
    # Convert BGR to RGB for compositing
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    composited_rgb = composite_with_background(
        result_rgb,
        refined_alpha,
        background_color_rgb
    )
    
    _debug_save(cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2BGR), "06_composited.png", job_id)
    
    # =========================================================================
    # Step 8: Place on biometric canvas (NO aggressive cropping!)
    # =========================================================================
    # CRITICAL: This step does NOT crop the image further.
    # It only scales and positions the PhotoRoom result on the canvas.
    # The portrait crop from Step 3 already includes shoulders.
    # We must preserve them here!
    print("\n[Step 8] Placing on biometric canvas (preserving shoulders)...")
    
    final_rgb = place_on_biometric_canvas(
        composited_rgb,
        face_center_in_crop,
        face_height_in_crop,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        background_color_rgb=background_color_rgb
    )
    
    # Convert to BGR for OpenCV
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    
    _debug_save(final_bgr, "07_final.png", job_id)
    
    # =========================================================================
    # Done
    # =========================================================================
    elapsed = time.time() - start_time
    
    metrics = {
        "success": True,
        "method": "photoroom_biometric",
        "input_size": f"{image_width}x{image_height}",
        "crop_box": list(crop_box),  # Convert tuple to list for JSON
        "output_size": f"{canvas_width}x{canvas_height}",
        "alpha_coverage_percent": float(round(alpha_coverage, 1)),
        "photoroom_metadata": photoroom_metadata,
        "processing_time_ms": float(round(elapsed * 1000, 1))
    }
    
    print(f"\n{'='*60}")
    print(f"[BiometricProcessor] Complete in {elapsed*1000:.0f}ms")
    print(f"  Output: {final_bgr.shape[1]}x{final_bgr.shape[0]}")
    print(f"{'='*60}\n")
    
    return final_bgr, metrics


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    import sys
    import glob
    
    os.environ["DEBUG_BG"] = "1"
    
    print("Biometric Processor - Standalone Test")
    print("=" * 50)
    
    # Find test image
    test_images = glob.glob("../uploads/*.jpg") + glob.glob("../uploads/*.jpeg")
    
    if not test_images:
        print("No test images found in uploads/")
        sys.exit(1)
    
    test_path = test_images[0]
    print(f"\nTesting with: {test_path}")
    
    # Load image
    image_bgr = cv2.imread(test_path)
    if image_bgr is None:
        print(f"Failed to load {test_path}")
        sys.exit(1)
    
    # Process
    output_bgr, metrics = process_biometric_photo(
        image_bgr,
        job_id="standalone_test"
    )
    
    # Save result
    output_path = "../outputs/biometric_test_result.jpg"
    cv2.imwrite(output_path, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"\nResult saved to: {output_path}")
    print(f"Metrics: {metrics}")

