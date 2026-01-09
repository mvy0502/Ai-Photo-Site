"""
Professional V2 Analyzer - MediaPipe Tasks API based
Uses FaceLandmarker for robust face analysis with landmarks, blendshapes, and iris detection.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import os

# MediaPipe Tasks API imports
MP_TASKS_AVAILABLE = False
python = None
vision = None
MPImage = None

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.core import image as mp_image_module
    MPImage = mp_image_module.Image
    ImageFormat = mp_image_module.ImageFormat
    MP_TASKS_AVAILABLE = True
except ImportError as e:
    MP_TASKS_AVAILABLE = False
    ImageFormat = None
    print(f"Warning: MediaPipe Tasks API not available: {e}")
    print("Falling back to v1 analyzer. Install MediaPipe 0.10+ for V2 analyzer.")

# Constants
POSE_YAW_MAX = 20.0
POSE_PITCH_MAX = 15.0
POSE_ROLL_MAX = 10.0
EYES_OPEN_MIN_PROB = 0.4
GAZE_FORWARD_MIN_PROB = 0.5
SUNGLASSES_PROB_THRESHOLD = 0.5
HAT_PROB_THRESHOLD = 0.5
HAIR_OVER_EYES_PROB_THRESHOLD = 0.4

# Model path (will be set if model file exists)
FACE_LANDMARKER_MODEL_PATH = None
# Try to find model file (any valid model file)
# Note: v2_with_blendshapes URL may not be available, so we accept basic model too
if os.path.exists("models/face_landmarker_v2_with_blendshapes.task"):
    # Check if file is valid (not a 404 error page)
    size = os.path.getsize("models/face_landmarker_v2_with_blendshapes.task")
    if size > 1000000:  # At least 1MB
        FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker_v2_with_blendshapes.task"
elif os.path.exists("models/face_landmarker.task"):
    size = os.path.getsize("models/face_landmarker.task")
    if size > 1000000:  # At least 1MB
        FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker.task"
elif os.path.exists("models/face_landmarker_v2.task"):
    size = os.path.getsize("models/face_landmarker_v2.task")
    if size > 1000000:  # At least 1MB
        FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker_v2.task"


class FaceLandmarkerAnalyzer:
    """MediaPipe Tasks API based face analyzer"""
    
    def __init__(self):
        self.landmarker = None
        self.initialized = False
        
        if not MP_TASKS_AVAILABLE:
            return
        
        if FACE_LANDMARKER_MODEL_PATH is None:
            print("⚠️  Warning: FaceLandmarker model file not found. Place model in models/ directory.")
            print("  Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_blendshapes/float16/1/face_landmarker_v2_with_blendshapes.task")
            return
        
        # Initialize FaceLandmarker
        try:
            base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
            
            # Try with blendshapes first (if model supports it)
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.initialized = True
                print("✅ [V2] FaceLandmarker initialized with blendshapes support")
                print(f"   Model: {FACE_LANDMARKER_MODEL_PATH}")
            except Exception as e1:
                # Fallback: without blendshapes (if model doesn't support it)
                if "blendshapes" in str(e1).lower() or "BLENDSHAPES" in str(e1):
                    print(f"⚠️  Blendshapes not available, trying without: {e1}")
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=True,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.landmarker = vision.FaceLandmarker.create_from_options(options)
                    self.initialized = True
                    print("✅ [V2] FaceLandmarker initialized without blendshapes (using EAR fallback)")
                    print(f"   Model: {FACE_LANDMARKER_MODEL_PATH}")
                else:
                    raise e1
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize FaceLandmarker: {e}")
            print("Falling back to v1 analyzer. Model file may be missing or corrupted.")
            import traceback
            traceback.print_exc()
            self.initialized = False
    
    def analyze(self, image_bgr: np.ndarray) -> Dict:
        """Analyze image and return comprehensive results"""
        if not self.initialized:
            return None
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # MediaPipe Tasks API Image creation
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=image_rgb)
            
            # Detect landmarks
            detection_result = self.landmarker.detect(mp_image)
            
            # Process results
            return self._process_detection_result(detection_result, image_bgr)
        except Exception as e:
            print(f"Error in FaceLandmarker analysis: {e}")
            return None
    
    def _process_detection_result(self, detection_result, image_bgr: np.ndarray) -> Dict:
        """Process MediaPipe detection results into our format"""
        h, w = image_bgr.shape[:2]
        
        # Face count
        face_count = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
        
        if face_count == 0:
            return {
                "face_count": 0,
                "face_detected": False,
                "error": "No face detected"
            }
        
        if face_count > 1:
            return {
                "face_count": face_count,
                "face_detected": True,
                "error": "Multiple faces detected"
            }
        
        # Get first (and only) face
        face_landmarks = detection_result.face_landmarks[0]
        face_blendshapes = detection_result.face_blendshapes[0] if detection_result.face_blendshapes else None
        transformation_matrix = detection_result.facial_transformation_matrixes[0] if detection_result.facial_transformation_matrixes else None
        
        # Extract key points
        landmarks_dict = self._extract_landmarks(face_landmarks, w, h)
        
        # Calculate metrics
        metrics = {
            "face_count": face_count,
            "face_detected": True,
        }
        
        # Head pose estimation
        pose = self._estimate_head_pose(landmarks_dict, transformation_matrix)
        if pose:
            metrics.update({
                "head_pose_yaw": pose["yaw"],
                "head_pose_pitch": pose["pitch"],
                "head_pose_roll": pose["roll"]
            })
        
        # Eyes open detection
        eyes_open_result = self._detect_eyes_open(landmarks_dict, face_blendshapes, image_bgr)
        if eyes_open_result:
            metrics.update({
                "eyes_open_prob": eyes_open_result["prob"],
                "eyes_open_method": eyes_open_result["method"]
            })
        
        # Gaze forward detection
        gaze_result = self._detect_gaze_forward(landmarks_dict, image_bgr)
        if gaze_result:
            metrics.update({
                "gaze_forward_prob": gaze_result["prob"],
                "gaze_score": gaze_result["score"]
            })
        
        # Glasses/Sunglasses detection (production-grade with reliability gate)
        # Use face detection confidence if available (default to 1.0)
        face_detection_confidence = 1.0  # Can be passed from detection result if available
        glasses_result = self._detect_glasses_sunglasses(landmarks_dict, image_bgr, face_detection_confidence)
        if glasses_result:
            metrics.update({
                # Legacy metrics
                "sunglasses_score": glasses_result["sunglasses_score"],
                "sunglasses_decision": glasses_result["sunglasses_decision"],
                "glasses_score": glasses_result["glasses_score"],
                "glasses_decision": glasses_result["glasses_decision"],
                "roi_reliability": glasses_result["roi_reliability"],
                "highlight_ratio": glasses_result["highlight_ratio"],
                "texture_energy": glasses_result["texture_energy"],
                "texture_energy_norm": glasses_result["texture_energy_norm"],
                "iris_visibility_proxy": glasses_result["iris_visibility_proxy"],
                "darkness_ratio": glasses_result["darkness_ratio"],
                "edge_density": glasses_result["edge_density"],
                # New lens-specific metrics for debugging
                "frame_presence_score": glasses_result.get("frame_presence_score", 0.0),
                "lens_dark_ratio": glasses_result.get("lens_dark_ratio", 0.0),
                "lens_mid_dark_ratio": glasses_result.get("lens_mid_dark_ratio", 0.0),
                "lens_p10_luma": glasses_result.get("lens_p10_luma", 100.0),
                "lens_mean_luma": glasses_result.get("lens_mean_luma", 128.0),
                "lens_area_ratio": glasses_result.get("lens_area_ratio", 0.0),
                "iris_visibility_lens": glasses_result.get("iris_visibility_lens", 0.5),
                "sunglasses_reason": glasses_result.get("sunglasses_reason", "none")
            })
            # Legacy compatibility
            metrics["iris_visibility"] = glasses_result["iris_visibility_proxy"]
        
        # Hat detection
        hat_result = self._detect_hat(landmarks_dict, image_bgr)
        if hat_result:
            metrics.update({
                "hat_score": hat_result["score"]
            })
        
        # Hair over eyes detection
        # Always calculate hair score, but analyze_v2.py will decide whether to use it
        # This allows for smarter logic: if hair score is high but sunglasses are also detected,
        # we can skip the hair issue
        hair_result = self._detect_hair_over_eyes(landmarks_dict, image_bgr)
        if hair_result:
            metrics.update({
                "hair_occlusion_score": hair_result["score"]
            })
        else:
            # If hair detection returned None, set to 0.0
            metrics.update({
                "hair_occlusion_score": 0.0
            })
        
        # Face bbox for crop
        face_bbox = self._calculate_face_bbox(landmarks_dict, w, h)
        
        return {
            "face_count": face_count,
            "face_detected": True,
            "metrics": metrics,
            "landmarks": landmarks_dict,
            "face_bbox": face_bbox,
            "blendshapes": face_blendshapes,
            "raw_landmarks": face_landmarks  # Raw MediaPipe landmarks for BG V2
        }
    
    def _extract_landmarks(self, landmarks, w: int, h: int) -> Dict:
        """Extract key landmark points from MediaPipe FaceLandmarker output"""
        result = {}
        
        # MediaPipe FaceLandmarker uses different landmark indices than Face Mesh
        # FaceLandmarker has 468 landmarks (same as Face Mesh)
        # Key indices (same as Face Mesh):
        # - Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # - Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        # - Nose tip: 4
        # - Chin: 18
        # - Forehead: 10
        # - Left iris: 468, 469, 470, 471, 472 (if available in v2)
        # - Right iris: 473, 474, 475, 476, 477 (if available in v2)
        
        num_landmarks = len(landmarks)
        
        # Eye landmarks
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        result["left_eye"] = [(landmarks[i].x * w, landmarks[i].y * h) 
                              for i in left_eye_indices if i < num_landmarks]
        result["right_eye"] = [(landmarks[i].x * w, landmarks[i].y * h) 
                               for i in right_eye_indices if i < num_landmarks]
        
        # Key points
        if num_landmarks > 4:
            result["nose_tip"] = (landmarks[4].x * w, landmarks[4].y * h)
        if num_landmarks > 18:
            result["chin"] = (landmarks[18].x * w, landmarks[18].y * h)
        if num_landmarks > 10:
            result["forehead"] = (landmarks[10].x * w, landmarks[10].y * h)
        
        # Iris centers (if available - MediaPipe face landmarker v2 has iris landmarks)
        if num_landmarks > 468:
            # Left iris center (approximate - use first iris landmark)
            left_iris_center = (landmarks[468].x * w, landmarks[468].y * h) if num_landmarks > 468 else None
            right_iris_center = (landmarks[473].x * w, landmarks[473].y * h) if num_landmarks > 473 else None
            if left_iris_center:
                result["left_iris_center"] = left_iris_center
            if right_iris_center:
                result["right_iris_center"] = right_iris_center
        
        return result
    
    def _estimate_head_pose(self, landmarks: Dict, transformation_matrix) -> Optional[Dict]:
        """Estimate head pose from landmarks and transformation matrix"""
        # Method 1: Use transformation matrix if available (more accurate)
        if transformation_matrix is not None:
            try:
                # MediaPipe transformation matrix is 4x4 numpy array or list
                import math
                # Convert to numpy array if needed
                if not isinstance(transformation_matrix, np.ndarray):
                    transformation_matrix = np.array(transformation_matrix)
                
                # Extract rotation matrix (3x3 top-left)
                R = transformation_matrix[:3, :3]
                
                # Extract Euler angles (yaw, pitch, roll)
                # Using ZYX convention (intrinsic rotations)
                yaw = math.atan2(R[1, 0], R[0, 0]) * 180 / math.pi
                pitch = math.asin(-R[2, 0]) * 180 / math.pi
                roll = math.atan2(R[2, 1], R[2, 2]) * 180 / math.pi
                
                return {
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "roll": float(roll)
                }
            except Exception as e:
                print(f"Error extracting pose from transformation matrix: {e}")
                # Fall through to landmark-based estimation
        
        # Method 2: Landmark-based pose estimation (simplified)
        if "nose_tip" in landmarks and "chin" in landmarks and "forehead" in landmarks:
            nose = landmarks["nose_tip"]
            chin = landmarks["chin"]
            forehead = landmarks["forehead"]
            
            # Calculate yaw (left/right rotation) - simplified heuristic
            # Compare nose position relative to face center
            face_center_x = (nose[0] + chin[0] + forehead[0]) / 3.0
            # Estimate yaw from horizontal deviation (simplified)
            # This is a placeholder - real implementation would use 3D landmarks
            yaw_estimate = (nose[0] - face_center_x) / 100.0  # Rough estimate
            
            # Estimate pitch from vertical positions
            face_center_y = (nose[1] + chin[1] + forehead[1]) / 3.0
            pitch_estimate = (nose[1] - face_center_y) / 100.0
            
            # Estimate roll from eye positions
            roll_estimate = 0.0
            if "left_eye" in landmarks and "right_eye" in landmarks:
                left_eye_y = np.mean([p[1] for p in landmarks["left_eye"]])
                right_eye_y = np.mean([p[1] for p in landmarks["right_eye"]])
                roll_estimate = (left_eye_y - right_eye_y) / 50.0  # Rough estimate
            
            return {
                "yaw": float(yaw_estimate * 10.0),  # Scale to degrees (rough)
                "pitch": float(pitch_estimate * 10.0),
                "roll": float(roll_estimate * 10.0)
            }
        
        return None
    
    def _detect_eyes_open(self, landmarks: Dict, blendshapes, image_bgr: np.ndarray) -> Optional[Dict]:
        """Detect if eyes are open using blendshapes (preferred) or EAR fallback"""
        # Method 1: Use blendshapes if available
        if blendshapes:
            for category in blendshapes:
                if category.category_name == "eyeBlinkLeft" or category.category_name == "eyeBlinkRight":
                    # Lower score = more open
                    blink_score = category.score
                    eyes_open_prob = 1.0 - blink_score
                    return {
                        "prob": eyes_open_prob,
                        "method": "blendshapes"
                    }
        
        # Method 2: EAR (Eye Aspect Ratio) fallback
        if "left_eye" in landmarks and "right_eye" in landmarks:
            left_ear = self._calculate_ear(landmarks["left_eye"])
            right_ear = self._calculate_ear(landmarks["right_eye"])
            
            if left_ear is not None and right_ear is not None:
                avg_ear = (left_ear + right_ear) / 2.0
                # EAR < 0.2 = closed, > 0.25 = open
                if avg_ear < 0.15:
                    eyes_open_prob = 0.0
                elif avg_ear > 0.25:
                    eyes_open_prob = 1.0
                else:
                    eyes_open_prob = (avg_ear - 0.15) / 0.10
                
                return {
                    "prob": max(0.0, min(1.0, eyes_open_prob)),
                    "method": "EAR"
                }
        
        return None
    
    def _calculate_ear(self, eye_points: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate Eye Aspect Ratio"""
        if len(eye_points) < 6:
            return None
        
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # Horizontal distance
        h1 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        if h1 == 0:
            return None
        
        ear = (v1 + v2) / (2.0 * h1)
        return ear
    
    def _detect_gaze_forward(self, landmarks: Dict, image_bgr: np.ndarray) -> Optional[Dict]:
        """Detect if gaze is forward using iris centers and eye corners"""
        h, w = image_bgr.shape[:2]
        
        # Method 1: Use iris centers if available
        if "left_iris_center" in landmarks and "right_iris_center" in landmarks:
            left_iris = landmarks["left_iris_center"]
            right_iris = landmarks["right_iris_center"]
            
            # Calculate eye centers
            if "left_eye" in landmarks and len(landmarks["left_eye"]) >= 4:
                left_eye_center = np.mean(landmarks["left_eye"], axis=0)
                left_deviation = np.linalg.norm(np.array(left_iris) - left_eye_center)
            else:
                left_deviation = 0.0
            
            if "right_eye" in landmarks and len(landmarks["right_eye"]) >= 4:
                right_eye_center = np.mean(landmarks["right_eye"], axis=0)
                right_deviation = np.linalg.norm(np.array(right_iris) - right_eye_center)
            else:
                right_deviation = 0.0
            
            # Average deviation
            avg_deviation = (left_deviation + right_deviation) / 2.0
            # Normalize (smaller deviation = more forward)
            gaze_score = 1.0 - min(1.0, avg_deviation / 20.0)  # 20px threshold
            gaze_prob = max(0.0, min(1.0, gaze_score))
            
            return {
                "prob": gaze_prob,
                "score": gaze_score
            }
        
        # Method 2: Face symmetry fallback
        if "nose_tip" in landmarks and "left_eye" in landmarks and "right_eye" in landmarks:
            nose = landmarks["nose_tip"]
            left_eye_center = np.mean(landmarks["left_eye"], axis=0)
            right_eye_center = np.mean(landmarks["right_eye"], axis=0)
            eye_center = (left_eye_center + right_eye_center) / 2.0
            
            # Check if nose is centered between eyes
            horizontal_deviation = abs(nose[0] - eye_center[0])
            gaze_score = 1.0 - min(1.0, horizontal_deviation / (w * 0.1))
            
            return {
                "prob": max(0.0, min(1.0, gaze_score)),
                "score": gaze_score
            }
        
        return None
    
    def _warp_eye_to_fixed_size(self, eye_points: List[Tuple[float, float]], 
                                  image_bgr: np.ndarray, target_w: int = 160, target_h: int = 80) -> Optional[np.ndarray]:
        """Perspective-warp eye polygon to fixed size"""
        if len(eye_points) < 4:
            return None
        
        # Convert to numpy array
        src_points = np.array(eye_points, dtype=np.float32)
        
        # Find bounding box
        x_coords = src_points[:, 0]
        y_coords = src_points[:, 1]
        x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
        y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
        
        # Create source rectangle (slightly expanded)
        margin = 5
        src_rect = np.array([
            [x_min - margin, y_min - margin],
            [x_max + margin, y_min - margin],
            [x_max + margin, y_max + margin],
            [x_min - margin, y_max + margin]
        ], dtype=np.float32)
        
        # Destination rectangle (fixed size)
        dst_rect = np.array([
            [0, 0],
            [target_w, 0],
            [target_w, target_h],
            [0, target_h]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(src_rect, dst_rect)
        
        # Warp
        warped = cv2.warpPerspective(image_bgr, M, (target_w, target_h))
        return warped
    
    def _compute_eye_metrics(self, eye_roi: np.ndarray) -> Dict:
        """Compute per-eye metrics for glasses/sunglasses detection"""
        if eye_roi is None or eye_roi.size == 0:
            return {
                "texture_energy": 0.0,
                "texture_energy_norm": 0.0,
                "highlight_ratio": 0.0,
                "edge_density": 0.0,
                "iris_visibility_proxy": 0.0,
                "darkness_ratio": 0.0,
                "roi_sharpness": 0.0
            }
        
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        area = h * w
        
        # a) texture_energy = variance(Laplacian(gray_roi))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_energy = float(np.var(laplacian))
        # Normalize texture_energy (typical range: 0-10000, normalize to 0-1)
        texture_energy_norm = min(1.0, texture_energy / 5000.0)
        
        # roi_sharpness (for reliability computation)
        roi_sharpness = texture_energy
        
        # b) highlight_ratio = count(gray_roi > 245) / area
        highlight_count = np.sum(gray > 245)
        highlight_ratio = float(highlight_count) / area if area > 0 else 0.0
        
        # c) darkness_ratio = count(gray_roi < 60) / area
        darkness_count = np.sum(gray < 60)
        darkness_ratio = float(darkness_count) / area if area > 0 else 0.0
        
        # d) edge_density = count(Canny(gray_roi)) / area
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.sum(edges > 0)
        edge_density = float(edge_count) / area if area > 0 else 0.0
        
        # e) iris_visibility_proxy: within central 40% of ROI, check edge_density
        # If edge_density exceeds minimum ring threshold, iris likely visible
        center_x, center_y = w // 2, h // 2
        center_w, center_h = int(w * 0.4), int(h * 0.4)
        x1, y1 = center_x - center_w // 2, center_y - center_h // 2
        x2, y2 = center_x + center_w // 2, center_y + center_h // 2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        center_roi = edges[y1:y2, x1:x2]
        center_area = (x2 - x1) * (y2 - y1)
        center_edge_density = float(np.sum(center_roi > 0)) / center_area if center_area > 0 else 0.0
        
        # Minimum ring threshold: if center edge density is low, iris likely occluded
        min_ring_threshold = 0.05  # 5% edge density in center
        iris_visibility_proxy = 1.0 if center_edge_density > min_ring_threshold else 0.0
        
        return {
            "texture_energy": texture_energy,
            "texture_energy_norm": texture_energy_norm,
            "highlight_ratio": highlight_ratio,
            "edge_density": edge_density,
            "iris_visibility_proxy": iris_visibility_proxy,
            "darkness_ratio": darkness_ratio,
            "roi_sharpness": roi_sharpness
        }
    
    def _sigmoid(self, x: float, k: float = 1.0) -> float:
        """Sigmoid function"""
        return 1.0 / (1.0 + np.exp(-k * x))
    
    def _compute_roi_reliability(self, left_eye_warped: np.ndarray, right_eye_warped: np.ndarray,
                                  left_metrics: Dict, right_metrics: Dict,
                                  landmarks: Dict, face_detection_confidence: float = 1.0) -> float:
        """
        Compute ROI reliability using sharpness, area, and landmark confidence.
        Returns: roi_reliability in [0, 1]
        """
        # 1. roi_sharpness (normalize Laplacian var)
        avg_sharpness = (left_metrics["roi_sharpness"] + right_metrics["roi_sharpness"]) / 2.0
        # Normalize: typical range 0-10000, normalize to 0-1
        sharpness_norm = min(1.0, avg_sharpness / 5000.0)
        
        # 2. eye_roi area (after warp) - normalize by expected area (160*80 = 12800)
        left_area = left_eye_warped.shape[0] * left_eye_warped.shape[1] if left_eye_warped is not None else 0
        right_area = right_eye_warped.shape[0] * right_eye_warped.shape[1] if right_eye_warped is not None else 0
        avg_area = (left_area + right_area) / 2.0
        expected_area = 160 * 80  # Fixed warp size
        area_norm = min(1.0, avg_area / expected_area)
        
        # 3. face_detection confidence / landmark presence
        # Check if we have good landmarks (both eyes present, reasonable number of points)
        has_left_eye = "left_eye" in landmarks and len(landmarks["left_eye"]) >= 8
        has_right_eye = "right_eye" in landmarks and len(landmarks["right_eye"]) >= 8
        landmark_presence = 1.0 if (has_left_eye and has_right_eye) else 0.5
        landmark_conf = face_detection_confidence * landmark_presence
        
        # Combine: roi_reliability = clamp(0.4*sharpness_norm + 0.3*area_norm + 0.3*landmark_conf, 0,1)
        roi_reliability = 0.4 * sharpness_norm + 0.3 * area_norm + 0.3 * landmark_conf
        roi_reliability = max(0.0, min(1.0, roi_reliability))
        
        return float(roi_reliability)
    
    def _compute_lens_only_metrics(self, eye_roi: np.ndarray) -> Dict:
        """
        Compute lens-only metrics by excluding frame edges.
        
        This analyzes the INNER portion of the eye ROI to detect lens tint
        without being fooled by dark glasses frames.
        
        Returns metrics specifically for lens darkness/tint detection.
        """
        if eye_roi is None or eye_roi.size == 0:
            return {
                "lens_dark_ratio": 0.0,
                "lens_mid_dark_ratio": 0.0,
                "lens_mean_luma": 128.0,
                "lens_p10_luma": 100.0,
                "lens_texture": 0.0,
                "lens_area_ratio": 0.0,
                "iris_visibility_lens": 0.5
            }
        
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        h, w = gray.shape
        total_area = h * w
        
        # Create lens mask by shrinking inward ~15% from edges (excludes frame)
        # This creates an elliptical central region
        center_x, center_y = w // 2, h // 2
        
        # Create mask for inner lens region (ellipse ~70% of ROI size)
        lens_mask = np.zeros((h, w), dtype=np.uint8)
        axes = (int(w * 0.35), int(h * 0.35))  # 70% of half-dimensions
        cv2.ellipse(lens_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        
        # Exclude specular highlights (very bright pixels > 245)
        highlight_mask = gray > 245
        lens_mask[highlight_mask] = 0
        
        # Get lens pixels
        lens_pixels = gray[lens_mask > 0]
        lens_area = len(lens_pixels)
        
        if lens_area < 100:  # Too few pixels for reliable analysis
            return {
                "lens_dark_ratio": 0.0,
                "lens_mid_dark_ratio": 0.0,
                "lens_mean_luma": 128.0,
                "lens_p10_luma": 100.0,
                "lens_texture": 0.0,
                "lens_area_ratio": 0.0,
                "iris_visibility_lens": 0.5
            }
        
        # Lens darkness metrics
        lens_dark_ratio = float(np.sum(lens_pixels < 70)) / lens_area
        lens_mid_dark_ratio = float(np.sum(lens_pixels < 100)) / lens_area
        lens_mean_luma = float(np.mean(lens_pixels))
        lens_p10_luma = float(np.percentile(lens_pixels, 10))  # 10th percentile (darker portion)
        
        # Lens texture (Laplacian variance inside lens region)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lens_texture = float(np.var(laplacian[lens_mask > 0]))
        
        # Lens area ratio (how much of ROI is usable lens area)
        lens_area_ratio = lens_area / total_area
        
        # Iris visibility inside lens region
        # Check for circular edge patterns in the center (iris ring)
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        if center_region.size > 0:
            center_edges = cv2.Canny(center_region, 30, 100)
            center_edge_density = float(np.sum(center_edges > 0)) / center_region.size
            # Higher edge density in center = more iris visibility
            iris_visibility_lens = min(1.0, center_edge_density * 20.0)
        else:
            iris_visibility_lens = 0.5
        
        return {
            "lens_dark_ratio": lens_dark_ratio,
            "lens_mid_dark_ratio": lens_mid_dark_ratio,
            "lens_mean_luma": lens_mean_luma,
            "lens_p10_luma": lens_p10_luma,
            "lens_texture": lens_texture,
            "lens_area_ratio": lens_area_ratio,
            "iris_visibility_lens": iris_visibility_lens
        }
    
    def _compute_frame_presence(self, eye_roi: np.ndarray) -> Dict:
        """
        Detect glasses frame presence (edges/structure around eyes).
        
        This is SEPARATE from lens tint detection.
        A frame can be present with clear lenses (regular glasses) or
        tinted lenses (sunglasses).
        
        Returns frame detection metrics.
        """
        if eye_roi is None or eye_roi.size == 0:
            return {
                "frame_presence_score": 0.0,
                "upper_edge_density": 0.0,
                "bridge_contrast": 0.0
            }
        
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        h, w = gray.shape
        
        # 1. Upper ROI band (top 25%) - where frame typically is
        upper_band = gray[:int(h * 0.25), :]
        if upper_band.size > 0:
            upper_edges = cv2.Canny(upper_band, 50, 150)
            upper_edge_density = float(np.sum(upper_edges > 0)) / upper_band.size
        else:
            upper_edge_density = 0.0
        
        # 2. Side bands (left and right 15%) - frame sides
        left_band = gray[:, :int(w * 0.15)]
        right_band = gray[:, int(w * 0.85):]
        
        side_edge_density = 0.0
        if left_band.size > 0 and right_band.size > 0:
            left_edges = cv2.Canny(left_band, 50, 150)
            right_edges = cv2.Canny(right_band, 50, 150)
            side_edge_density = (float(np.sum(left_edges > 0)) / left_band.size + 
                                float(np.sum(right_edges > 0)) / right_band.size) / 2.0
        
        # 3. Bridge area contrast (between eyes - usually has horizontal frame line)
        # This is approximated by looking at the center-top region
        bridge_region = gray[:int(h * 0.3), int(w * 0.35):int(w * 0.65)]
        if bridge_region.size > 0:
            # Look for horizontal edges (glasses bridge)
            sobel_x = cv2.Sobel(bridge_region, cv2.CV_64F, 1, 0, ksize=3)
            bridge_contrast = float(np.std(sobel_x))
        else:
            bridge_contrast = 0.0
        
        # Combine into frame presence score
        # High upper edge density + side edges = frame likely present
        frame_presence_score = (
            0.5 * min(1.0, upper_edge_density * 8.0) +  # Upper frame
            0.3 * min(1.0, side_edge_density * 8.0) +   # Side frames
            0.2 * min(1.0, bridge_contrast / 30.0)       # Bridge
        )
        
        return {
            "frame_presence_score": float(frame_presence_score),
            "upper_edge_density": float(upper_edge_density),
            "bridge_contrast": float(bridge_contrast)
        }
    
    def _detect_glasses_sunglasses(self, landmarks: Dict, image_bgr: np.ndarray, 
                                    face_detection_confidence: float = 1.0) -> Optional[Dict]:
        """
        Glasses/Sunglasses detection with SEPARATE frame vs lens tint analysis.
        
        KEY PRINCIPLE:
        - SUNGLASSES = dark/tinted LENS (blocks iris visibility)
        - GLASSES = frame present with CLEAR lens
        - Frame edges should NOT trigger sunglasses detection
        
        Decision logic:
        1. Compute lens-only metrics (excluding frame)
        2. Compute frame presence metrics
        3. SUNGLASSES fail only if lens is actually dark AND iris not visible
        4. GLASSES warn if frame is present but lens is clear
        
        Returns all metrics for debugging and decision making.
        """
        default_result = {
            "sunglasses_score": 0.0,
            "sunglasses_decision": "pass",
            "glasses_score": 0.0,
            "glasses_decision": "pass",
            "roi_reliability": 0.0,
            "highlight_ratio": 0.0,
            "texture_energy": 0.0,
            "texture_energy_norm": 0.0,
            "iris_visibility_proxy": 0.0,
            "darkness_ratio": 0.0,
            "edge_density": 0.0,
            # New lens-specific metrics
            "frame_presence_score": 0.0,
            "lens_dark_ratio": 0.0,
            "lens_mid_dark_ratio": 0.0,
            "lens_p10_luma": 100.0,
            "lens_mean_luma": 128.0,
            "lens_area_ratio": 0.0,
            "iris_visibility_lens": 0.5,
            "sunglasses_reason": "none"
        }
        
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return default_result
        
        # Build eye polygons from landmarks
        left_eye_points = landmarks["left_eye"]
        right_eye_points = landmarks["right_eye"]
        
        # Warp each eye to fixed size (160x80)
        left_eye_warped = self._warp_eye_to_fixed_size(left_eye_points, image_bgr)
        right_eye_warped = self._warp_eye_to_fixed_size(right_eye_points, image_bgr)
        
        if left_eye_warped is None or right_eye_warped is None:
            return default_result
        
        # =====================================================================
        # Step 1: Compute standard eye metrics (for backwards compatibility)
        # =====================================================================
        left_metrics = self._compute_eye_metrics(left_eye_warped)
        right_metrics = self._compute_eye_metrics(right_eye_warped)
        
        avg_texture_energy = (left_metrics["texture_energy"] + right_metrics["texture_energy"]) / 2.0
        avg_texture_energy_norm = (left_metrics["texture_energy_norm"] + right_metrics["texture_energy_norm"]) / 2.0
        avg_highlight_ratio = (left_metrics["highlight_ratio"] + right_metrics["highlight_ratio"]) / 2.0
        avg_edge_density = (left_metrics["edge_density"] + right_metrics["edge_density"]) / 2.0
        avg_iris_visibility = (left_metrics["iris_visibility_proxy"] + right_metrics["iris_visibility_proxy"]) / 2.0
        avg_darkness_ratio = (left_metrics["darkness_ratio"] + right_metrics["darkness_ratio"]) / 2.0
        
        roi_reliability = self._compute_roi_reliability(
            left_eye_warped, right_eye_warped,
            left_metrics, right_metrics,
            landmarks, face_detection_confidence
        )
        
        # =====================================================================
        # Step 2: Compute LENS-ONLY metrics (excludes frame)
        # =====================================================================
        left_lens = self._compute_lens_only_metrics(left_eye_warped)
        right_lens = self._compute_lens_only_metrics(right_eye_warped)
        
        # Average lens metrics
        lens_dark_ratio = (left_lens["lens_dark_ratio"] + right_lens["lens_dark_ratio"]) / 2.0
        lens_mid_dark_ratio = (left_lens["lens_mid_dark_ratio"] + right_lens["lens_mid_dark_ratio"]) / 2.0
        lens_mean_luma = (left_lens["lens_mean_luma"] + right_lens["lens_mean_luma"]) / 2.0
        lens_p10_luma = (left_lens["lens_p10_luma"] + right_lens["lens_p10_luma"]) / 2.0
        lens_area_ratio = (left_lens["lens_area_ratio"] + right_lens["lens_area_ratio"]) / 2.0
        iris_visibility_lens = (left_lens["iris_visibility_lens"] + right_lens["iris_visibility_lens"]) / 2.0
        
        # =====================================================================
        # Step 3: Compute FRAME presence metrics
        # =====================================================================
        left_frame = self._compute_frame_presence(left_eye_warped)
        right_frame = self._compute_frame_presence(right_eye_warped)
        
        frame_presence_score = (left_frame["frame_presence_score"] + right_frame["frame_presence_score"]) / 2.0
        
        # =====================================================================
        # Step 4: SUNGLASSES decision (based on LENS tint, NOT frame)
        # =====================================================================
        # SUNGLASSES requires BOTH:
        # - Dark lens (high lens_dark_ratio, low p10_luma)
        # - Low iris visibility
        # Frame edges alone should NOT trigger sunglasses
        
        sunglasses_decision = "pass"
        sunglasses_reason = "none"
        
        # Deterministic rule-based decision
        # FAIL: Truly tinted lens with no iris visibility
        if (lens_dark_ratio > 0.35 and 
            lens_p10_luma < 75 and 
            iris_visibility_lens < 0.35):
            sunglasses_decision = "fail"
            sunglasses_reason = f"dark_lens(dark={lens_dark_ratio:.2f},p10={lens_p10_luma:.0f},iris={iris_visibility_lens:.2f})"
        
        # WARN: Moderately tinted or suspicious
        elif (lens_dark_ratio > 0.22 and iris_visibility_lens < 0.45):
            sunglasses_decision = "warn"
            sunglasses_reason = f"moderate_tint(dark={lens_dark_ratio:.2f},iris={iris_visibility_lens:.2f})"
        
        # Additional check: very low mean luma in lens region
        elif (lens_mean_luma < 60 and iris_visibility_lens < 0.40):
            sunglasses_decision = "warn"
            sunglasses_reason = f"low_luma(mean={lens_mean_luma:.0f},iris={iris_visibility_lens:.2f})"
        
        # Safety check: If lens area is too small, don't fail for sunglasses
        if lens_area_ratio < 0.15 and sunglasses_decision == "fail":
            sunglasses_decision = "warn"
            sunglasses_reason += "_degraded_small_lens_area"
        
        # Low reliability gate: never fail, only warn
        if roi_reliability < 0.50 and sunglasses_decision == "fail":
            sunglasses_decision = "warn"
            sunglasses_reason += "_degraded_low_reliability"
        
        # Compute sunglasses_score for backwards compatibility
        # This is now derived from lens metrics, not edge density
        sunglasses_score = (
            0.4 * lens_dark_ratio +
            0.3 * (1.0 - iris_visibility_lens) +
            0.2 * (1.0 - min(1.0, lens_p10_luma / 150.0)) +
            0.1 * lens_mid_dark_ratio
        )
        sunglasses_score = max(0.0, min(1.0, sunglasses_score))
        
        # =====================================================================
        # Step 5: GLASSES decision (based on FRAME presence, NOT lens tint)
        # =====================================================================
        # GLASSES is detected if frame is present but sunglasses is NOT triggered
        
        glasses_decision = "pass"
        glasses_score = frame_presence_score
        
        # Only emit GLASSES if sunglasses not triggered
        if sunglasses_decision == "pass":
            if frame_presence_score >= 0.55:
                glasses_decision = "warn"  # GLASSES is always warn, never fail
            # Also check for high upper edge density as frame indicator
            elif avg_edge_density > 0.08 and avg_highlight_ratio > 0.01:
                # Frame edges + reflections = likely glasses
                glasses_decision = "warn"
                glasses_score = max(glasses_score, 0.55)
        
        # =====================================================================
        # Return all metrics
        # =====================================================================
        return {
            "sunglasses_score": float(sunglasses_score),
            "sunglasses_decision": sunglasses_decision,
            "glasses_score": float(glasses_score),
            "glasses_decision": glasses_decision,
            "roi_reliability": float(roi_reliability),
            "highlight_ratio": float(avg_highlight_ratio),
            "texture_energy": float(avg_texture_energy),
            "texture_energy_norm": float(avg_texture_energy_norm),
            "iris_visibility_proxy": float(avg_iris_visibility),
            "darkness_ratio": float(avg_darkness_ratio),
            "edge_density": float(avg_edge_density),
            # New lens-specific metrics for debugging
            "frame_presence_score": float(frame_presence_score),
            "lens_dark_ratio": float(lens_dark_ratio),
            "lens_mid_dark_ratio": float(lens_mid_dark_ratio),
            "lens_p10_luma": float(lens_p10_luma),
            "lens_mean_luma": float(lens_mean_luma),
            "lens_area_ratio": float(lens_area_ratio),
            "iris_visibility_lens": float(iris_visibility_lens),
            "sunglasses_reason": sunglasses_reason
        }
    
    def _detect_iris_visibility(self, landmarks: Dict, image_bgr: np.ndarray) -> Optional[Dict]:
        """Legacy method - now uses _detect_glasses_sunglasses"""
        result = self._detect_glasses_sunglasses(landmarks, image_bgr)
        if result:
            # Map to old format for backward compatibility
            return {
                "visibility": result["iris_visibility_proxy"],
                "sunglasses_score": result["sunglasses_score"]
            }
        return {
            "visibility": 0.5,
            "sunglasses_score": 0.0
        }
    
    def _extract_eye_roi(self, eye_points: List[Tuple[float, float]], image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract eye region of interest"""
        if len(eye_points) < 4:
            return None
        
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        
        x_min = int(max(0, min(x_coords) - 10))
        x_max = int(min(image_bgr.shape[1], max(x_coords) + 10))
        y_min = int(max(0, min(y_coords) - 10))
        y_max = int(min(image_bgr.shape[0], max(y_coords) + 10))
        
        if x_max > x_min and y_max > y_min:
            return image_bgr[y_min:y_max, x_min:x_max]
        return None
    
    def _detect_hat(self, landmarks: Dict, image_bgr: np.ndarray) -> Optional[Dict]:
        """Detect hat using forehead ROI and hairline analysis"""
        h, w = image_bgr.shape[:2]
        
        if "forehead" not in landmarks or "left_eye" not in landmarks:
            return None
        
        # Forehead region (above eyes, extending upward)
        forehead_y = landmarks["forehead"][1]
        eye_y = np.mean([p[1] for p in landmarks["left_eye"]])
        
        # Extract forehead ROI
        forehead_top = max(0, int(forehead_y - h * 0.15))
        forehead_bottom = int(eye_y)
        forehead_left = max(0, int(w * 0.2))
        forehead_right = min(w, int(w * 0.8))
        
        if forehead_bottom > forehead_top and forehead_right > forehead_left:
            forehead_roi = image_bgr[forehead_top:forehead_bottom, forehead_left:forehead_right]
            gray_forehead = cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2GRAY)
            
            # Analyze variance and edge patterns
            variance = np.var(gray_forehead)
            mean_brightness = np.mean(gray_forehead)
            
            # Check for hat edge (horizontal line)
            edges = cv2.Canny(gray_forehead, 50, 150)
            horizontal_edges = cv2.Sobel(gray_forehead, cv2.CV_64F, 0, 1, ksize=3)
            horizontal_edge_strength = np.mean(np.abs(horizontal_edges))
            
            # Score calculation
            hat_score = 0.0
            if variance < 300:
                hat_score += 0.3
            if variance < 200 and (mean_brightness < 80 or mean_brightness > 200):
                hat_score += 0.4
            if horizontal_edge_strength > 20:
                hat_score += 0.3
            
            return {
                "score": min(1.0, hat_score)
            }
        
        return None
    
    def _detect_hair_over_eyes(self, landmarks: Dict, image_bgr: np.ndarray) -> Optional[Dict]:
        """Detect hair occlusion over eyes (NOT sunglasses - that's handled separately)"""
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return None
        
        # Extract eye ROIs
        left_eye_roi = self._extract_eye_roi(landmarks["left_eye"], image_bgr)
        right_eye_roi = self._extract_eye_roi(landmarks["right_eye"], image_bgr)
        
        if left_eye_roi is None or right_eye_roi is None:
            return None
        
        # Analyze occlusion (dark regions that might be hair)
        left_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Check for dark regions (hair occlusion)
        left_dark = np.sum(left_gray < 50) / left_gray.size
        right_dark = np.sum(right_gray < 50) / right_gray.size
        avg_dark_ratio = (left_dark + right_dark) / 2.0
        
        # Check for texture patterns (hair has different texture than skin/glasses)
        left_std = np.std(left_gray)
        right_std = np.std(right_gray)
        avg_std = (left_std + right_std) / 2.0
        
        # IMPORTANT: Distinguish between sunglasses and hair
        # Sunglasses: very uniform dark (low std, high dark ratio)
        # Hair: more textured dark (higher std, high dark ratio)
        # If it looks like sunglasses (very uniform), don't count as hair
        # More aggressive: if very dark and uniform, definitely sunglasses
        # CRITICAL: Check avg_mean as well (not just dark ratio)
        avg_mean = (np.mean(left_gray) + np.mean(right_gray)) / 2.0
        
        # If very dark (low mean) and uniform (low std), it's sunglasses
        # VERY AGGRESSIVE: Lower thresholds to catch all sunglasses cases
        if avg_mean < 80 and avg_std < 20:
            # Dark and uniform = definitely sunglasses
            return {
                "score": 0.0  # Let sunglasses detection handle this
            }
        if avg_mean < 100 and avg_std < 25:
            # Somewhat dark and uniform = likely sunglasses
            return {
                "score": 0.0  # Let sunglasses detection handle this
            }
        if avg_dark_ratio > 0.20 and avg_std < 12:
            # High dark ratio and low std = sunglasses
            return {
                "score": 0.0  # Let sunglasses detection handle this
            }
        if avg_dark_ratio > 0.15 and avg_std < 15:
            # This looks like sunglasses, not hair - return low score
            return {
                "score": 0.0  # Let sunglasses detection handle this
            }
        if avg_mean < 100 and avg_dark_ratio > 0.10 and avg_std < 10:
            # Dark mean + high dark ratio + very uniform = sunglasses
            return {
                "score": 0.0  # Let sunglasses detection handle this
            }
        
        # Score: high dark ratio + medium/high std = likely hair occlusion
        # STRICTER LOGIC: Require more evidence before flagging hair
        # Normal hair that doesn't actually cover eyes should not trigger
        occlusion_score = 0.0
        
        # Only flag if VERY dark (> 25% dark pixels) AND textured (hair)
        if avg_dark_ratio > 0.25 and avg_std > 20:
            occlusion_score += 0.5
        
        # Additional score only if extremely dark (> 35% dark pixels)
        if avg_dark_ratio > 0.35 and avg_std > 25:
            occlusion_score += 0.5
        
        return {
            "score": min(1.0, occlusion_score)
        }
    
    def _calculate_face_bbox(self, landmarks: Dict, w: int, h: int) -> Tuple[float, float, float, float]:
        """Calculate face bounding box from landmarks (relative coordinates)"""
        # Collect all landmark points
        all_points = []
        if "left_eye" in landmarks:
            all_points.extend(landmarks["left_eye"])
        if "right_eye" in landmarks:
            all_points.extend(landmarks["right_eye"])
        if "nose_tip" in landmarks:
            all_points.append(landmarks["nose_tip"])
        if "chin" in landmarks:
            all_points.append(landmarks["chin"])
        if "forehead" in landmarks:
            all_points.append(landmarks["forehead"])
        
        if not all_points:
            return (0.25, 0.25, 0.5, 0.5)  # Default
        
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        x_min = min(x_coords) / w
        y_min = min(y_coords) / h
        x_max = max(x_coords) / w
        y_max = max(y_coords) / h
        
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)


# Global analyzer instance
_analyzer_instance = None

def get_analyzer() -> Optional[FaceLandmarkerAnalyzer]:
    """Get or create analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FaceLandmarkerAnalyzer()
    return _analyzer_instance if _analyzer_instance.initialized else None

