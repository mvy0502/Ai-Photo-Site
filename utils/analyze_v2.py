"""
Biometric Photo Analysis V2 - Complete Pipeline
Contract-based response with priority ladder and proper gating.
"""

import cv2
import numpy as np
import os
import uuid
from typing import Dict, Optional, List, Any
from .analyzer_v2 import get_analyzer

# =============================================================================
# CONFIGURATION
# =============================================================================
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"

# Face Size Thresholds
FACE_TOO_LARGE_AREA_RATIO = 0.50      # face_area / image_area
FACE_TOO_LARGE_HEIGHT_RATIO = 0.82   # face_height / image_height
FACE_TOO_SMALL_AREA_RATIO = 0.05     # Minimum face area (lowered from 0.08)
FACE_TOO_SMALL_HEIGHT_RATIO = 0.25   # Minimum face height (lowered from 0.35)
FACE_CROPPED_MARGIN_RATIO = 0.02     # Minimum edge margin

# Head Pose Thresholds
HEAD_POSE_YAW_MAX = 15.0
HEAD_POSE_PITCH_MAX = 15.0
HEAD_POSE_ROLL_MAX = 10.0

# Blur Threshold
BLUR_THRESHOLD = 50.0
BLUR_RELIABILITY_GATE = 0.5

# Eyewear Thresholds
SUNGLASSES_SCORE_THRESHOLD = 0.70
IRIS_VISIBILITY_SUNGLASSES = 0.35    # Below this = sunglasses
IRIS_VISIBILITY_GLASSES = 0.85       # Below this = glasses

# Expression Thresholds
MAR_THRESHOLD = 0.15                 # Mouth Aspect Ratio
TEETH_BRIGHT_RATIO = 0.15

# Hair Thresholds (very strict to avoid false positives)
HAIR_DARK_RATIO_THRESHOLD = 0.70
HAIR_EDGE_DENSITY_THRESHOLD = 0.30
EAR_GATING_THRESHOLD = 0.20          # Eye Aspect Ratio

# Accessory Thresholds - headphones are DARK + SMOOTH
ACCESSORY_DARK_RATIO_THRESHOLD = 0.45  # Dark threshold
ACCESSORY_EDGE_MAX_FOR_HEADPHONES = 0.08  # Headphones are smooth (low edges)
# Busy backgrounds have high edges (> 0.10) so they won't trigger


# =============================================================================
# ISSUE FACTORY
# =============================================================================
def create_issue(
    issue_id: str,
    severity: str,
    title: str,
    message: str,
    title_tr: str,
    message_tr: str,
    confidence: float = 1.0,
    requires_ack: bool = False,
    debug_metrics: Optional[Dict] = None
) -> Dict:
    """Create a standardized issue object."""
    return {
        "id": issue_id,
        "severity": severity,
        "title": title,
        "message": message,
        "title_tr": title_tr,
        "message_tr": message_tr,
        "confidence": round(confidence, 2),
        "requires_ack": requires_ack,
        "debug": {"metrics": debug_metrics or {}}
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================
def analyze_image_v2(
    job_id: str,
    file_path: str,
    thresholds: Optional[Dict] = None,
    acknowledged_ids: Optional[List[str]] = None
) -> Dict:
    """
    Professional V2 biometric photo analysis.
    
    Returns contract-compliant response:
    {
        "final_status": "PASS|WARN|FAIL",
        "server_can_continue": bool,
        "requires_acknowledgement": bool,
        "required_ack_issue_ids": [...],
        "issues": [...],
        "metrics": {...},
        "preview_image": "ORIGINAL_URL",
        "debug_overlay_image": "ONLY_IF_DEV_MODE"
    }
    """
    print(f"🔵 [V2] Analyzing {job_id}")
    
    if acknowledged_ids is None:
        acknowledged_ids = []
    if acknowledged_ids:
        print(f"🔵 [V2] Acknowledged: {acknowledged_ids}")
    
    # ==========================================================================
    # STEP 1: Load Image (IMMUTABLE - never modify original)
    # ==========================================================================
    original_image = cv2.imread(file_path)
    if original_image is None:
        return _error_response("LOAD_ERROR", "Dosya yüklenemedi", 
                               "Fotoğraf dosyası açılamadı. Lütfen farklı bir dosya deneyin.")
    
    h, w = original_image.shape[:2]
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    issues = []
    metrics = {"image_width": w, "image_height": h}
    
    # ==========================================================================
    # STEP 2: MediaPipe Analysis
    # ==========================================================================
    analyzer = get_analyzer()
    if not analyzer:
        return _error_response("ANALYZER_ERROR", "Analiz hatası",
                               "Yüz analizi yapılamadı. Lütfen tekrar deneyin.")
    
    mp_result = analyzer.analyze(original_image)
    if not mp_result:
        return _error_response("ANALYZER_ERROR", "Analiz hatası",
                               "Yüz analizi yapılamadı. Lütfen tekrar deneyin.")
    
    mp_metrics = mp_result.get("metrics", {})
    raw_landmarks = mp_result.get("raw_landmarks")
    face_bbox = mp_result.get("face_bbox")
    face_count = mp_result.get("face_count", 0)
    
    # ==========================================================================
    # PRIORITY 1: Face Detection (NO_FACE / MULTIPLE_FACES)
    # ==========================================================================
    if face_count == 0:
        issues.append(create_issue(
            "NO_FACE", "FAIL",
            "No face detected", "Could not detect a face in the photo.",
            "Yüz bulunamadı", "Fotoğrafta yüz tespit edilemedi. Lütfen yüzünüzün net göründüğü bir fotoğraf yükleyin."
        ))
        return _build_response(issues, metrics, file_path, acknowledged_ids)
    
    if face_count > 1:
        issues.append(create_issue(
            "MULTIPLE_FACES", "FAIL",
            "Multiple faces", "Multiple faces detected in the photo.",
            "Birden fazla yüz", "Fotoğrafta birden fazla kişi var. Lütfen tek kişilik fotoğraf yükleyin."
        ))
        return _build_response(issues, metrics, file_path, acknowledged_ids)
    
    # ==========================================================================
    # PRIORITY 2: Face Size - FACE_TOO_LARGE Early Exit
    # ==========================================================================
    if face_bbox:
        bbox_x, bbox_y, bbox_w, bbox_h = face_bbox
        face_area = bbox_w * bbox_h
        image_area = 1.0  # Normalized
        
        face_area_ratio = face_area
        face_height_ratio = bbox_h
        
        metrics["face_bbox_area_ratio"] = round(face_area_ratio, 3)
        metrics["face_bbox_height_ratio"] = round(face_height_ratio, 3)
        
        # FACE_TOO_LARGE - Early exit, suppress all other checks
        if face_area_ratio > FACE_TOO_LARGE_AREA_RATIO or face_height_ratio > FACE_TOO_LARGE_HEIGHT_RATIO:
            issues.append(create_issue(
                "FACE_TOO_LARGE", "FAIL",
                "Photo too close", "Face appears too large in the photo.",
                "Fotoğraf çok yakın", "Yüz fotoğrafta çok büyük görünüyor. Kameradan biraz uzaklaşın ve tekrar çekin.",
                debug_metrics={"face_area_ratio": face_area_ratio, "face_height_ratio": face_height_ratio}
            ))
            print(f"🔵 [V2] ❌ FACE_TOO_LARGE - Early exit (area={face_area_ratio:.2f}, height={face_height_ratio:.2f})")
            return _build_response(issues, metrics, file_path, acknowledged_ids)
        
        # FACE_CROPPED - Early exit
        margin_left = bbox_x
        margin_right = 1.0 - (bbox_x + bbox_w)
        margin_top = bbox_y
        margin_bottom = 1.0 - (bbox_y + bbox_h)
        min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
        
        if min_margin < FACE_CROPPED_MARGIN_RATIO:
            issues.append(create_issue(
                "FACE_CROPPED", "FAIL",
                "Face cropped", "Face is too close to the edge of the photo.",
                "Yüz kenardan kırpılmış", "Yüz fotoğrafın kenarına çok yakın. Yüzün tam görünmesini sağlayın.",
                debug_metrics={"min_margin": min_margin}
            ))
            print(f"🔵 [V2] ❌ FACE_CROPPED - Early exit (min_margin={min_margin:.3f})")
            return _build_response(issues, metrics, file_path, acknowledged_ids)
        
        # FACE_TOO_SMALL
        if face_area_ratio < FACE_TOO_SMALL_AREA_RATIO or face_height_ratio < FACE_TOO_SMALL_HEIGHT_RATIO:
            issues.append(create_issue(
                "FACE_TOO_SMALL", "FAIL",
                "Face too small", "Face appears too small in the photo.",
                "Yüz çok küçük", "Yüz fotoğrafta çok küçük görünüyor. Kameraya biraz yaklaşın.",
                debug_metrics={"face_area_ratio": face_area_ratio, "face_height_ratio": face_height_ratio}
            ))
    
    # ==========================================================================
    # PRIORITY 3: Eyewear (SUNGLASSES = FAIL, GLASSES = WARN)
    # ==========================================================================
    sunglasses_score = mp_metrics.get("sunglasses_score", 0.0)
    iris_visibility = mp_metrics.get("iris_visibility_lens", mp_metrics.get("iris_visibility_proxy", 1.0))
    
    metrics["eyewear_sunglasses_score"] = round(sunglasses_score, 2)
    metrics["eyewear_iris_visibility"] = round(iris_visibility, 2)
    
    eyewear_decision = "NONE"
    has_sunglasses_fail = False
    has_glasses_warn = False
    
    if sunglasses_score > SUNGLASSES_SCORE_THRESHOLD and iris_visibility < IRIS_VISIBILITY_SUNGLASSES:
        eyewear_decision = "SUNGLASSES"
        has_sunglasses_fail = True
        issues.append(create_issue(
            "EYEWEAR_SUNGLASSES", "FAIL",
            "Sunglasses detected", "Dark lenses are not allowed in biometric photos.",
            "Güneş gözlüğü / koyu lens tespit edildi", 
            "Biyometrik fotoğraflarda genellikle kabul edilmez. Lütfen gözlüğü çıkarıp tekrar yükleyin.",
            confidence=sunglasses_score,
            debug_metrics={"sunglasses_score": sunglasses_score, "iris_visibility": iris_visibility}
        ))
        print(f"🔵 [V2] ❌ SUNGLASSES FAIL")
    
    elif iris_visibility < IRIS_VISIBILITY_GLASSES and sunglasses_score > 0.55:
        eyewear_decision = "GLASSES"
        has_glasses_warn = True
        issues.append(create_issue(
            "EYEWEAR_GLASSES", "WARN",
            "Glasses detected", "Some institutions may not accept photos with glasses.",
            "Gözlük tespit edildi",
            "Bazı ülkelerde/kurumlarda gözlükle biyometrik fotoğraf kabul edilmeyebilir. Kabul ihtimalini artırmak için çıkarmanızı öneririz.",
            confidence=1.0 - iris_visibility,
            requires_ack=True,
            debug_metrics={"iris_visibility": iris_visibility, "sunglasses_score": sunglasses_score}
        ))
        print(f"🔵 [V2] ⚠️ GLASSES WARN (iris_vis={iris_visibility:.2f})")
    
    metrics["eyewear_decision"] = eyewear_decision
    
    # ==========================================================================
    # PRIORITY 4: Head Pose
    # ==========================================================================
    yaw = abs(mp_metrics.get("head_pose_yaw", 0.0))
    pitch = abs(mp_metrics.get("head_pose_pitch", 0.0))
    roll = abs(mp_metrics.get("head_pose_roll", 0.0))
    
    metrics["head_pose_yaw"] = round(yaw, 1)
    metrics["head_pose_pitch"] = round(pitch, 1)
    metrics["head_pose_roll"] = round(roll, 1)
    
    if yaw > HEAD_POSE_YAW_MAX:
        issues.append(create_issue(
            "HEAD_POSE_YAW", "FAIL",
            "Head turned sideways", "Head is turned too far left or right.",
            "Baş yana dönük", "Başınız çok fazla sağa veya sola dönük. Düz bakın.",
            debug_metrics={"yaw": yaw, "threshold": HEAD_POSE_YAW_MAX}
        ))
    
    if pitch > HEAD_POSE_PITCH_MAX:
        issues.append(create_issue(
            "HEAD_POSE_PITCH", "FAIL",
            "Head tilted up/down", "Head is tilted too far up or down.",
            "Baş yukarı/aşağı eğik", "Başınız çok fazla yukarı veya aşağı eğik. Düz bakın.",
            debug_metrics={"pitch": pitch, "threshold": HEAD_POSE_PITCH_MAX}
        ))
    
    if roll > HEAD_POSE_ROLL_MAX:
        issues.append(create_issue(
            "HEAD_POSE_ROLL", "FAIL",
            "Head tilted", "Head is tilted to one side.",
            "Baş eğik", "Başınız yana eğik. Düz tutun.",
            debug_metrics={"roll": roll, "threshold": HEAD_POSE_ROLL_MAX}
        ))
    
    # ==========================================================================
    # PRIORITY 5: Blur (with reliability gating)
    # ==========================================================================
    face_reliability = mp_metrics.get("roi_reliability", 1.0)
    metrics["face_reliability"] = round(face_reliability, 2)
    
    if face_bbox and face_reliability >= BLUR_RELIABILITY_GATE:
        x = int(face_bbox[0] * w)
        y = int(face_bbox[1] * h)
        fw = int(face_bbox[2] * w)
        fh = int(face_bbox[3] * h)
        
        face_roi = gray[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)]
        if face_roi.size > 0:
            blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            metrics["blur_score"] = round(blur_score, 2)
            
            if blur_score < BLUR_THRESHOLD:
                issues.append(create_issue(
                    "BLURRED", "FAIL",
                    "Photo blurry", "The photo is too blurry.",
                    "Bulanık fotoğraf", "Fotoğraf çok bulanık. Daha net bir fotoğraf çekin.",
                    debug_metrics={"blur_score": blur_score, "threshold": BLUR_THRESHOLD}
                ))
    
    # ==========================================================================
    # PRIORITY 6: Eyes Open / Gaze
    # ==========================================================================
    eyes_open_prob = mp_metrics.get("eyes_open_prob", 1.0)
    gaze_forward_prob = mp_metrics.get("gaze_forward_prob", 1.0)
    
    metrics["eyes_open_prob"] = round(eyes_open_prob, 2)
    metrics["gaze_forward_prob"] = round(gaze_forward_prob, 2)
    
    has_eyes_closed = False
    if eyes_open_prob < 0.5:
        has_eyes_closed = True
        issues.append(create_issue(
            "EYES_CLOSED", "FAIL",
            "Eyes closed", "Eyes must be open.",
            "Gözler kapalı", "Gözleriniz açık olmalı.",
            debug_metrics={"eyes_open_prob": eyes_open_prob}
        ))
    
    if gaze_forward_prob < 0.5:
        issues.append(create_issue(
            "NOT_LOOKING_AT_CAMERA", "FAIL",
            "Not looking at camera", "Please look directly at the camera.",
            "Kameraya bakmıyor", "Lütfen doğrudan kameraya bakın.",
            debug_metrics={"gaze_forward_prob": gaze_forward_prob}
        ))
    
    # ==========================================================================
    # PRIORITY 7: Expression (Teeth/Mouth Open)
    # Suppress if sunglasses FAIL (priority ladder)
    # ==========================================================================
    has_expression_fail = False
    
    if not has_sunglasses_fail and raw_landmarks and len(raw_landmarks) > 0:
        try:
            # Calculate MAR (Mouth Aspect Ratio)
            upper_lip_idx, lower_lip_idx = 13, 14
            mouth_left_idx, mouth_right_idx = 61, 291
            
            if all(idx < len(raw_landmarks) for idx in [upper_lip_idx, lower_lip_idx, mouth_left_idx, mouth_right_idx]):
                upper_y = raw_landmarks[upper_lip_idx].y
                lower_y = raw_landmarks[lower_lip_idx].y
                left_x = raw_landmarks[mouth_left_idx].x
                right_x = raw_landmarks[mouth_right_idx].x
                
                vertical_dist = abs(lower_y - upper_y)
                horizontal_dist = abs(right_x - left_x)
                
                if horizontal_dist > 0:
                    mar = vertical_dist / horizontal_dist
                    metrics["mouth_mar"] = round(mar, 3)
                    
                    if mar > MAR_THRESHOLD:
                        has_expression_fail = True
                        issues.append(create_issue(
                            "TEETH_VISIBLE", "FAIL",
                            "Teeth visible / Mouth open", "Please close your mouth.",
                            "Dişler görünüyor / Ağız açık",
                            "Biyometrik fotoğraflarda dişler görünmemeli. Ağzınızı kapatın.",
                            debug_metrics={"mar": mar, "threshold": MAR_THRESHOLD}
                        ))
                        print(f"🔵 [V2] ❌ TEETH_VISIBLE (MAR={mar:.3f})")
        except Exception as e:
            print(f"🔵 [V2] ⚠️ MAR calculation error: {e}")
    
    # ==========================================================================
    # PRIORITY 8: Hair Over Eyes
    # Suppress if: expression fail, sunglasses fail, glasses warn, eyes closed
    # ==========================================================================
    hair_gating_ok = (
        not has_expression_fail and
        not has_sunglasses_fail and
        not has_glasses_warn and
        not has_eyes_closed and
        eyes_open_prob >= 0.60
    )
    
    if hair_gating_ok and raw_landmarks and len(raw_landmarks) > 0:
        try:
            hair_metrics = _detect_hair_over_eyes(gray, raw_landmarks, w, h)
            if hair_metrics:
                metrics["hair_detection"] = hair_metrics
                
                # Very strict: both eyes must show strong signal
                left_signal = (hair_metrics.get("left_dark_ratio", 0) > HAIR_DARK_RATIO_THRESHOLD and
                              hair_metrics.get("left_edge_density", 0) > HAIR_EDGE_DENSITY_THRESHOLD)
                right_signal = (hair_metrics.get("right_dark_ratio", 0) > HAIR_DARK_RATIO_THRESHOLD and
                               hair_metrics.get("right_edge_density", 0) > HAIR_EDGE_DENSITY_THRESHOLD)
                
                if left_signal and right_signal:
                    issues.append(create_issue(
                        "HAIR_OVER_EYES", "FAIL",
                        "Hair covering eyes", "Hair should not cover the eyes.",
                        "Saç gözleri kapatıyor", "Gözleriniz tamamen görünür olmalı.",
                        debug_metrics=hair_metrics
                    ))
                    print(f"🔵 [V2] ❌ HAIR_OVER_EYES detected")
        except Exception as e:
            print(f"🔵 [V2] ⚠️ Hair detection error: {e}")
    else:
        if has_expression_fail:
            print(f"🔵 [V2] ⚠️ Hair check suppressed (expression FAIL)")
        elif has_glasses_warn:
            print(f"🔵 [V2] ⚠️ Hair check suppressed (glasses WARN)")
    
    # ==========================================================================
    # PRIORITY 9: Accessory Detection (Headphones, etc.)
    # Suppress if expression fail
    # ==========================================================================
    if not has_expression_fail and raw_landmarks and len(raw_landmarks) > 0:
        try:
            accessory_metrics = _detect_accessories(gray, raw_landmarks, w, h, mp_metrics)
            if accessory_metrics:
                metrics["accessory_detection"] = accessory_metrics
                
                if accessory_metrics.get("headphones_detected", False):
                    issues.append(create_issue(
                        "ACCESSORY_HEADPHONES", "FAIL",
                        "Headphones detected", "Please remove headphones.",
                        "Kulaklık tespit edildi", "Lütfen kulaklığı çıkarın.",
                        debug_metrics=accessory_metrics
                    ))
                    print(f"🔵 [V2] ❌ HEADPHONES detected")
        except Exception as e:
            print(f"🔵 [V2] ⚠️ Accessory detection error: {e}")
    elif has_expression_fail:
        print(f"🔵 [V2] ⚠️ Accessory check suppressed (expression FAIL)")
    
    # ==========================================================================
    # BUILD RESPONSE
    # ==========================================================================
    return _build_response(issues, metrics, file_path, acknowledged_ids)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _error_response(issue_id: str, title_tr: str, message_tr: str) -> Dict:
    """Create error response."""
    issues = [create_issue(
        issue_id, "FAIL",
        title_tr, message_tr,  # Using TR as EN fallback
        title_tr, message_tr
    )]
    return {
        "status": "done",
        "final_status": "FAIL",
        "server_can_continue": False,
        "requires_acknowledgement": False,
        "required_ack_issue_ids": [],
        "issues": issues,
        "metrics": {},
        "preview_image": None,
        "debug_overlay_image": None,
        # Legacy fields
        "can_continue": False,
        "ok": False,
        "result": "fail",
        "reasons": [message_tr],
        "fix_plan": []
    }


def _build_response(
    issues: List[Dict],
    metrics: Dict,
    original_file_path: str,
    acknowledged_ids: List[str]
) -> Dict:
    """Build the contract-compliant response."""
    
    # Determine final_status
    fail_issues = [i for i in issues if i["severity"] == "FAIL"]
    warn_issues = [i for i in issues if i["severity"] == "WARN"]
    
    if fail_issues:
        final_status = "FAIL"
    elif warn_issues:
        final_status = "WARN"
    else:
        final_status = "PASS"
    
    # Determine requires_acknowledgement and required_ack_issue_ids
    required_ack_issue_ids = [
        i["id"] for i in issues
        if i["severity"] == "WARN" and i.get("requires_ack", False) and i["id"] not in acknowledged_ids
    ]
    requires_acknowledgement = len(required_ack_issue_ids) > 0
    
    # Determine server_can_continue
    # False if any FAIL, or if WARN requires ack and not acknowledged
    if final_status == "FAIL":
        server_can_continue = False
    elif requires_acknowledgement:
        server_can_continue = False
    else:
        server_can_continue = True
    
    print(f"🔵 [V2] final_status={final_status}, server_can_continue={server_can_continue}, required_ack={required_ack_issue_ids}")
    
    # Build checks object for legacy frontend compatibility
    # Derive from issues - if no FACE_NOT_FOUND issue, face was detected
    issue_ids = [i["id"] for i in issues]
    checks = {
        "face_detected": "FACE_NOT_FOUND" not in issue_ids,
        "single_face": "MULTIPLE_FACES" not in issue_ids,
        "min_size": "FACE_TOO_SMALL" not in issue_ids,
        "aspect_ratio_ok": "FACE_CROPPED" not in issue_ids and "FACE_TOO_LARGE" not in issue_ids
    }
    
    # Build preview_url from original_file_path
    preview_url = None
    if original_file_path:
        # Convert file path to URL path
        if original_file_path.startswith("uploads/"):
            preview_url = "/" + original_file_path
        elif "/uploads/" in original_file_path:
            preview_url = original_file_path[original_file_path.index("/uploads/"):]
        else:
            preview_url = "/uploads/" + os.path.basename(original_file_path)
    
    response = {
        "status": "done",
        "final_status": final_status,
        "server_can_continue": server_can_continue,
        "requires_acknowledgement": requires_acknowledgement,
        "required_ack_issue_ids": required_ack_issue_ids,
        "issues": issues,
        "metrics": metrics,
        "checks": checks,  # Legacy checklist items
        "preview_url": preview_url,  # URL for frontend preview
        "preview_image": original_file_path,  # ALWAYS original
        "debug_overlay_image": None,  # Only set in DEV_MODE
        # Legacy fields for backward compatibility
        "can_continue": server_can_continue,
        "pending_ack_ids": required_ack_issue_ids,
        "ok": server_can_continue,
        "result": final_status.lower(),
        "reasons": [i["message_tr"] for i in fail_issues],
        "fix_plan": [] if final_status == "FAIL" else ["background_replace_white", "crop_to_tr_biometric_50x60"]
    }
    
    return response


def _detect_hair_over_eyes(gray: np.ndarray, raw_landmarks: List, w: int, h: int) -> Optional[Dict]:
    """Detect hair covering eyes using sclera-only mask."""
    try:
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        left_iris_idx, right_iris_idx = 468, 473
        
        def get_eye_metrics(eye_indices, iris_idx, cheek_idx):
            eye_points = [(int(raw_landmarks[i].x * w), int(raw_landmarks[i].y * h))
                         for i in eye_indices if i < len(raw_landmarks)]
            if len(eye_points) < 6:
                return None
            
            eye_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(eye_mask, [np.array(eye_points, dtype=np.int32)], 255)
            
            # Get iris position
            if iris_idx < len(raw_landmarks):
                iris_cx = int(raw_landmarks[iris_idx].x * w)
                iris_cy = int(raw_landmarks[iris_idx].y * h)
            else:
                eye_ys, eye_xs = np.where(eye_mask > 0)
                if len(eye_xs) == 0:
                    return None
                iris_cx, iris_cy = int(np.mean(eye_xs)), int(np.mean(eye_ys))
            
            # Calculate iris radius and create sclera mask
            eye_ys, eye_xs = np.where(eye_mask > 0)
            if len(eye_xs) == 0:
                return None
            eye_width = np.max(eye_xs) - np.min(eye_xs)
            iris_radius = int(eye_width * 0.35)
            
            iris_mask = np.zeros_like(eye_mask)
            cv2.circle(iris_mask, (iris_cx, iris_cy), iris_radius, 255, -1)
            sclera_mask = cv2.bitwise_and(eye_mask, cv2.bitwise_not(iris_mask))
            
            # Use top 50% of sclera
            sclera_ys, sclera_xs = np.where(sclera_mask > 0)
            if len(sclera_ys) < 20:
                return None
            y_median = np.median(sclera_ys)
            sclera_top_mask = sclera_mask.copy()
            sclera_top_mask[int(y_median):, :] = 0
            
            sclera_pixels = gray[sclera_top_mask > 0]
            if len(sclera_pixels) < 10:
                return None
            
            # Get skin baseline
            skin_luma = 128
            if cheek_idx < len(raw_landmarks):
                cx = int(raw_landmarks[cheek_idx].x * w)
                cy = int(raw_landmarks[cheek_idx].y * h)
                patch = gray[max(0,cy-10):min(h,cy+10), max(0,cx-10):min(w,cx+10)]
                if patch.size > 0:
                    skin_luma = np.mean(patch)
            
            dark_thresh = skin_luma * 0.5
            dark_ratio = np.sum(sclera_pixels < dark_thresh) / len(sclera_pixels)
            
            sclera_roi = cv2.bitwise_and(gray, gray, mask=sclera_top_mask)
            edges = cv2.Canny(sclera_roi, 50, 150)
            edge_density = np.sum(edges > 0) / len(sclera_pixels)
            
            return {"dark_ratio": float(dark_ratio), "edge_density": float(edge_density)}
        
        left_metrics = get_eye_metrics(left_eye_indices, left_iris_idx, 50)
        right_metrics = get_eye_metrics(right_eye_indices, right_iris_idx, 280)
        
        if left_metrics and right_metrics:
            return {
                "left_dark_ratio": float(round(left_metrics["dark_ratio"], 2)),
                "left_edge_density": float(round(left_metrics["edge_density"], 2)),
                "right_dark_ratio": float(round(right_metrics["dark_ratio"], 2)),
                "right_edge_density": float(round(right_metrics["edge_density"], 2))
            }
    except Exception as e:
        print(f"🔵 [V2] Hair detection error: {e}")
    
    return None


def _detect_accessories(gray: np.ndarray, raw_landmarks: List, w: int, h: int, mp_metrics: Dict) -> Optional[Dict]:
    """Detect accessories like headphones using band-based analysis.
    
    Key improvements to avoid false positives:
    1. Higher dark ratio threshold (0.65 instead of 0.50)
    2. Edge density check - headphones have smooth surfaces (low edge density)
    3. Busy backgrounds have high edge density (lots of texture)
    """
    try:
        # Method 1: MediaPipe hat_score (only if very confident)
        hat_score = mp_metrics.get("hat_score", 0.0)
        if hat_score > 0.50:  # Increased from 0.35
            return {"headphones_detected": True, "method": "hat_score", "score": hat_score}
        
        # Method 2: Band-based detection with edge analysis
        forehead_idx, chin_idx = 10, 152
        left_temple_idx, right_temple_idx = 234, 454
        
        if not all(idx < len(raw_landmarks) for idx in [forehead_idx, chin_idx, left_temple_idx, right_temple_idx]):
            return None
        
        forehead_y = raw_landmarks[forehead_idx].y * h
        chin_y = raw_landmarks[chin_idx].y * h
        left_x = raw_landmarks[left_temple_idx].x * w
        right_x = raw_landmarks[right_temple_idx].x * w
        face_height = chin_y - forehead_y
        face_width = right_x - left_x
        
        head_x_min = max(0, int(left_x - 0.12 * face_width))
        head_x_max = min(w, int(right_x + 0.12 * face_width))
        eyebrow_y = int(forehead_y + 0.15 * face_height)
        
        left_band = gray[eyebrow_y:int(chin_y), head_x_min:int(left_x)]
        right_band = gray[eyebrow_y:int(chin_y), int(right_x):head_x_max]
        
        def analyze_band(band, dark_thresh=80):
            """Analyze a band for darkness AND edge density."""
            if band.size < 50:
                return {"dark_ratio": 0.0, "edge_density": 0.0}
            
            dark_ratio = float(np.sum(band < dark_thresh) / band.size)
            
            # Compute edge density - busy backgrounds have HIGH edges
            edges = cv2.Canny(band, 50, 150)
            edge_density = float(np.sum(edges > 0) / band.size)
            
            return {"dark_ratio": dark_ratio, "edge_density": edge_density}
        
        left_analysis = analyze_band(left_band)
        right_analysis = analyze_band(right_band)
        
        left_dark = left_analysis["dark_ratio"]
        right_dark = right_analysis["dark_ratio"]
        left_edges = left_analysis["edge_density"]
        right_edges = right_analysis["edge_density"]
        
        # Headphones detection criteria:
        # 1. Both sides must be dark
        # 2. Symmetric darkness  
        # 3. LOW edge density (headphones are SMOOTH, backgrounds are TEXTURED)
        symmetry = abs(left_dark - right_dark) < 0.20
        both_dark = left_dark > ACCESSORY_DARK_RATIO_THRESHOLD and right_dark > ACCESSORY_DARK_RATIO_THRESHOLD
        
        # Key insight: Headphones are SMOOTH (low edges), busy backgrounds are TEXTURED (high edges)
        # Headphones: edge_density < 0.08 (smooth surface)
        # Background: edge_density > 0.10 (lots of texture)
        left_smooth = left_edges < ACCESSORY_EDGE_MAX_FOR_HEADPHONES
        right_smooth = right_edges < ACCESSORY_EDGE_MAX_FOR_HEADPHONES
        both_smooth = left_smooth and right_smooth
        
        # Only trigger if DARK + SMOOTH + SYMMETRIC
        is_headphones = both_dark and symmetry and both_smooth
        
        result = {
            "headphones_detected": bool(is_headphones),  # Convert numpy bool to Python bool
            "left_dark_ratio": float(round(left_dark, 2)),
            "right_dark_ratio": float(round(right_dark, 2)),
            "left_edge_density": float(round(left_edges, 3)),
            "right_edge_density": float(round(right_edges, 3)),
            "both_dark": bool(both_dark),
            "both_smooth": bool(both_smooth),
            "symmetric": bool(symmetry)
        }
        
        if is_headphones:
            result["method"] = "band_analysis"
        
        return result
    except Exception as e:
        print(f"🔵 [V2] Accessory detection error: {e}")
    
    return None
