from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import cv2
import numpy as np
import mediapipe as mp
import threading
import stripe
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# V2 Analyzer - New contract-based analysis
from utils.analyze_v2 import analyze_image_v2
from utils.biometric_processor import process_biometric_photo
from utils.photoroom_client import check_api_configuration

# Payment and Email modules
from utils.payment import (
    create_checkout_session, verify_webhook_signature, handle_checkout_completed,
    is_paid, get_order, create_order, update_order_state,
    generate_signed_url, verify_signed_url, generate_email_link,
    get_price_display, get_stripe_publishable_key,
    STRIPE_WEBHOOK_SECRET,
    is_payments_enabled, get_payments_config, PaymentsDisabledError
)
from utils.email_service import send_download_link, is_email_configured
from utils.db import db_manager
from utils.env_config import get_config_summary, startup_validation
from utils.supabase_storage import (
    is_storage_configured, upload_bytes, create_signed_url, download_bytes,
    get_object_key, get_content_type, validate_image_bytes,
    upload_bytes_sync, create_signed_url_sync, download_bytes_sync
)
from utils.db_jobs import (
    create_job_safe, get_job_safe, update_job_safe,
    save_analysis_result, save_processed_image, update_job,
    JobStatus, PaymentState, DatabaseError,
    DEV_ALLOW_MEMORY_FALLBACK
)

# Feature flag for V2 analyzer
USE_V2_ANALYZER = True
PIPELINE_VERSION = "2.1.0-photoroom"

def check_photoroom_configured() -> bool:
    """Check if PhotoRoom API is configured."""
    try:
        config = check_api_configuration()
        return config.get("api_configured", False)
    except:
        return False

# ============================================================================
# Threshold Configuration - Kolay ayarlanabilir değerler
# ============================================================================
# FAIL kriterleri (AI ile düzeltilemez)
FACE_BLUR_THRESHOLD = 50.0      # Face ROI Laplacian variance (yüz bulanıklığı)
FACE_BRIGHTNESS_MIN = 50.0      # Face ROI minimum brightness (yüz karanlık)
FACE_BRIGHTNESS_MAX = 240.0     # Face ROI maximum brightness (yüz parlak + clipping kontrolü)
FACE_RATIO_MIN_UNRECOVERABLE = 0.05  # Yüz çok küçük (kurtarılamaz)
FACE_RATIO_MAX_UNRECOVERABLE = 0.60  # Yüz çok büyük (kurtarılamaz)
MIN_RESOLUTION = 400 * 400      # Minimum çözünürlük (width * height)

# Otomatik düzeltilebilir (FAIL vermez)
FACE_RATIO_MIN_RECOVERABLE = 0.06   # Yüz küçük ama kurtarılabilir
FACE_RATIO_MAX_RECOVERABLE = 0.50   # Yüz büyük ama kurtarılabilir

FACE_DETECTION_ENABLED = True  # Face detection aktif/pasif

# Checklist kontrolleri için
MIN_SHORT_SIDE = 400  # Minimum kısa kenar (pixel)
# ============================================================================

app = FastAPI()

# ============================================================================
# DATABASE STARTUP & HEALTH
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and validate configuration on startup."""
    # Run config validation first
    startup_validation()
    
    # Initialize database
    success, message = await db_manager.initialize()
    if success:
        print(f"✅ Database: {message}")
    else:
        print(f"⚠️ Database: {message}")
        print("   App will continue without database. Some features may be unavailable.")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown."""
    await db_manager.close()
    print("🔌 Database connections closed.")

@app.get("/api/health", response_class=JSONResponse)
async def health():
    """
    Basic health check endpoint for load balancers and uptime monitoring.
    Returns minimal info without exposing internals.
    """
    return JSONResponse({
        "status": "ok",
        "service": "biyometrikfoto-api",
        "version": PIPELINE_VERSION
    })


@app.get("/api/health/db", response_class=JSONResponse)
async def health_db():
    """
    Database health check endpoint.
    Executes SELECT 1 and returns connection status.
    
    Returns:
    - ok: bool - Whether SELECT 1 succeeded
    - db: int - Result of SELECT 1 (should be 1)
    - connection_info: dict - URL type, port, and prepared statement cache status
    """
    if not db_manager.connection_info.get("connected"):
        return JSONResponse({
            "ok": False,
            "error": "Database not connected",
            "connection_info": {
                "url_type": db_manager.connection_info.get("url_type", "unknown"),
                "port": db_manager.connection_info.get("port"),
                "prepared_statements_disabled": db_manager.connection_info.get("prepared_statements_disabled", False)
            }
        }, status_code=503)
    
    try:
        success, result = await db_manager.test_connection()
        return JSONResponse({
            "ok": True,
            "db": result,
            "connection_info": {
                "url_type": db_manager.connection_info.get("url_type"),
                "port": db_manager.connection_info.get("port"),
                "prepared_statements_disabled": db_manager.connection_info.get("prepared_statements_disabled", False)
            }
        })
    except Exception as e:
        # Sanitize error message
        import re
        error_msg = re.sub(r'://[^@]+@', '://***:***@', str(e))
        return JSONResponse({
            "ok": False,
            "error": error_msg[:200],
            "connection_info": db_manager.connection_info
        }, status_code=503)

# ============================================================================
# JOB STATUS HELPERS
# ============================================================================

# In-memory cache for active processing jobs (cleared after analysis)
# Only used during the brief processing window, NOT for reads
_processing_jobs: dict[str, dict] = {}


def _format_job_response(db_job: dict) -> dict:
    """
    Convert DB job record to API response format.
    Maintains backward compatibility with frontend expectations.
    """
    analysis_result = db_job.get("analysis_result") or {}
    
    # Reconstruct preview URL from job ID
    job_id = str(db_job.get("id", ""))
    original_path = db_job.get("original_image_path", "")
    
    # Use /api/preview/{job_id} endpoint which handles both Supabase and local storage
    # This endpoint returns a redirect to either a signed URL or local file
    preview_url = f"/api/preview/{job_id}" if job_id else None
    
    response = {
        # Core status
        "status": "done" if db_job.get("status") in [JobStatus.PASS, JobStatus.WARN, JobStatus.FAIL] else db_job.get("status", "processing").lower(),
        "final_status": analysis_result.get("final_status", db_job.get("status")),
        
        # Issues and acknowledgement
        "issues": analysis_result.get("issues", []),
        "can_continue": db_job.get("can_continue", False),
        "server_can_continue": db_job.get("can_continue", False),
        "requires_ack_ids": db_job.get("requires_ack_ids", []),
        "acknowledged_issue_ids": db_job.get("acknowledged_issue_ids", []),
        
        # Image paths
        "preview_url": preview_url,
        "preview_image": preview_url,
        "normalized_url": db_job.get("normalized_image_path"),
        "processed_url": db_job.get("processed_image_path"),
        
        # Payment state
        "payment_state": db_job.get("payment_state", PaymentState.ANALYZED),
        
        # Merge in other analysis fields
        **{k: v for k, v in analysis_result.items() 
           if k not in ["issues", "final_status", "can_continue", "server_can_continue"]}
    }
    
    return response

# Templates klasörünü ayarla
templates = Jinja2Templates(directory="templates")

# ============================================================================
# Create runtime directories (Render fresh container needs these)
# ============================================================================
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Static dosyaları mount et
app.mount("/static", StaticFiles(directory="static"), name="static")

# Uploads klasörünü serve et (preview için)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Maksimum dosya boyutu (8MB)
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB in bytes

# MediaPipe Face Detection - lazy load (analyze_image içinde yüklenecek)

def get_file_extension(filename: str) -> str:
    """Dosya uzantısını döndürür"""
    return os.path.splitext(filename)[1].lower()

def is_allowed_file(filename: str) -> bool:
    """Dosya uzantısının izin verilenler arasında olup olmadığını kontrol eder"""
    ext = get_file_extension(filename)
    return ext in ALLOWED_EXTENSIONS

def analyze_image(job_id: str, file_path: str) -> dict:
    """Gerçek görüntü analizi - Ürün felsefesi: Sadece düzeltilemez durumlar FAIL"""
    try:
        # 1) Görüntüyü oku
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            return {
                "status": "done",
                "result": "fail",
                "reasons": ["Görüntü okunamadı"],
                "fix_plan": [],
                "metrics": {}
            }
        
        height, width = image_bgr.shape[:2]
        image_area = width * height
        
        # 2) Min çözünürlük kontrolü
        if image_area < MIN_RESOLUTION:
            return {
                "status": "done",
                "result": "fail",
                "reasons": ["Fotoğraf çözünürlüğü çok düşük"],
                "fix_plan": [],
                "metrics": {"resolution": f"{width}x{height}"}
            }
        
        # 3) Face detection (mediapipe -> haar fallback)
        face_count = None
        face_area_ratio = None
        face_backend = "none"
        face_bbox = None
        
        if FACE_DETECTION_ENABLED:
            # MediaPipe
            try:
                import mediapipe as mp
                mp_face = mp.solutions.face_detection
                with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)
                
                faces = []
                if results and results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        faces.append((bbox.xmin, bbox.ymin, bbox.width, bbox.height))
                
                face_count = len(faces)
                face_backend = "mediapipe"
                
                if face_count > 0:
                    largest = max(faces, key=lambda b: b[2] * b[3])
                    face_area_ratio = float(largest[2] * largest[3])
                    face_bbox = largest
            except Exception:
                pass
            
            # OpenCV Haar fallback
            if face_count is None:
                try:
                    gray_temp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    detected = face_cascade.detectMultiScale(gray_temp, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    
                    face_count = len(detected)
                    face_backend = "opencv_haar"
                    
                    if face_count > 0:
                        h, w = gray_temp.shape[:2]
                        x, y, fw, fh = max(detected, key=lambda r: r[2] * r[3])
                        face_area_ratio = float((fw * fh) / (w * h))
                        face_bbox = (x / w, y / h, fw / w, fh / h)
                except Exception:
                    pass
        
        # Face detection unavailable
        if face_count is None:
            face_count = -1
            face_backend = "unavailable"
            face_area_ratio = None
            face_bbox = None
        
        # 4) Face ROI üzerinden analiz (blur, brightness, highlight clipping)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_roi_blur = None
        face_roi_brightness = None
        face_roi_highlight_clipping = None
        
        if face_count == 1 and face_bbox is not None:
            h, w = gray.shape[:2]
            bbox_x, bbox_y, bbox_w, bbox_h = face_bbox
            x = int(bbox_x * w)
            y = int(bbox_y * h)
            fw = int(bbox_w * w)
            fh = int(bbox_h * h)
            
            y_start = max(0, y)
            y_end = min(h, y + fh)
            x_start = max(0, x)
            x_end = min(w, x + fw)
            
            if y_end > y_start and x_end > x_start:
                face_roi = gray[y_start:y_end, x_start:x_end]
                face_roi_blur = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                face_roi_brightness = float(np.mean(face_roi))
                # Highlight clipping kontrolü (255'e yakın pixel yüzdesi)
                highlight_pixels = np.sum(face_roi >= 250)
                highlight_ratio = highlight_pixels / face_roi.size
                face_roi_highlight_clipping = highlight_ratio > 0.1  # %10'dan fazla clipping
        
        # 5) FAIL kriterleri (sadece düzeltilemez durumlar)
        reasons = []
        fix_plan = []
        
        # Face detection unavailable ise yüz kontrollerini skip et
        if face_backend != "unavailable" and face_count != -1:
            # FAIL 1: Yüz yok
            if face_count == 0:
                reasons.append("Yüz tespit edilemedi")
            
            # FAIL 2: Birden fazla yüz
            elif face_count > 1:
                reasons.append("Fotoğrafta birden fazla kişi var")
            
            # FAIL 3, 4, 5, 6: Tek yüz varsa kontroller
            elif face_count == 1:
                # FAIL 3: Yüz aşırı bulanık
                if face_roi_blur is not None and face_roi_blur < FACE_BLUR_THRESHOLD:
                    reasons.append("Fotoğraf çok bulanık (yüz net değil)")
                
                # FAIL 4: Yüz çok karanlık
                if face_roi_brightness is not None and face_roi_brightness < FACE_BRIGHTNESS_MIN:
                    reasons.append("Yüz çok karanlık")
                
                # FAIL 5: Yüz çok parlak VE detay kaybı
                if face_roi_brightness is not None and face_roi_brightness > FACE_BRIGHTNESS_MAX:
                    if face_roi_highlight_clipping:
                        reasons.append("Yüz aşırı parlak (detay kaybı)")
                
                # FAIL 6: Yüz aşırı küçük/büyük (kurtarılamaz)
                if face_area_ratio is not None:
                    if face_area_ratio < FACE_RATIO_MIN_UNRECOVERABLE:
                        reasons.append("Yüz kadrajı uygun değil")
                    elif face_area_ratio > FACE_RATIO_MAX_UNRECOVERABLE:
                        reasons.append("Yüz kadrajı uygun değil")
        
        # 6) Otomatik düzeltilebilir durumlar (FAIL vermez, fix_plan'a ekle)
        if len(reasons) == 0:  # FAIL yoksa fix_plan doldur
            # Arka plan kontrolü (basit - edge detection ile)
            fix_plan.append("background_replace_white")
            
            # Oran kontrolü
            fix_plan.append("crop_to_tr_biometric_50x60")
            
            # Işık dengesi
            fix_plan.append("auto_exposure_balance")
            
            # Kafa eğimi (face detection varsa)
            if face_count == 1:
                fix_plan.append("straighten_head_tilt")
            
            # Yüz kadrajı (kurtarılabilir aralıkta ise)
            if face_area_ratio is not None:
                if FACE_RATIO_MIN_RECOVERABLE <= face_area_ratio <= FACE_RATIO_MAX_RECOVERABLE:
                    if face_area_ratio < FACE_RATIO_MIN_UNRECOVERABLE or face_area_ratio > FACE_RATIO_MAX_UNRECOVERABLE:
                        fix_plan.append("smart_crop_face_centered")
        
        # 7) Sonuç
        result = "pass" if len(reasons) == 0 else "fail"
        
        # 8) Checks (checklist için)
        checks = {
            "face_detected": face_count >= 1 if face_count is not None and face_count != -1 else None,
            "single_face": face_count == 1 if face_count is not None and face_count != -1 else None,
            "min_size": min(width, height) >= MIN_SHORT_SIDE,
            "aspect_ratio_ok": True  # Şimdilik her zaman true (crop ile düzelteceğiz)
        }
        
        # 9) Overlay (50x60mm frame + face bbox)
        overlay = {
            "frame_width_mm": 50,
            "frame_height_mm": 60,
            "face_height_mm": None,
            "bbox_rel": None
        }
        
        if face_bbox is not None:
            overlay["bbox_rel"] = {
                "xmin": float(face_bbox[0]),
                "ymin": float(face_bbox[1]),
                "width": float(face_bbox[2]),
                "height": float(face_bbox[3])
            }
            # face_height_mm: yüz bbox yüksekliği oranını 60mm ile çarp
            overlay["face_height_mm"] = float(face_bbox[3] * 60)
        
        # 10) Metrics (debug için)
        metrics = {
            "face_detection_backend": face_backend,
            "resolution": f"{width}x{height}",
            "global_blur": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)
        }
        
        if face_count is not None and face_count != -1:
            metrics["faces"] = face_count
        if face_area_ratio is not None:
            metrics["face_ratio"] = round(float(face_area_ratio), 4)
        if face_roi_blur is not None:
            metrics["face_blur"] = round(float(face_roi_blur), 2)
        if face_roi_brightness is not None:
            metrics["face_brightness"] = round(float(face_roi_brightness), 2)
        
        return {
            "status": "done",
            "result": result,
            "reasons": reasons,
            "fix_plan": fix_plan,
            "checks": checks,
            "overlay": overlay,
            "metrics": metrics
        }
    except Exception as e:
        return {
            "status": "done",
            "result": "fail",
            "reasons": [f"Analiz hatası: {str(e)}"],
            "fix_plan": [],
            "checks": {
                "face_detected": None,
                "single_face": None,
                "min_size": None,
                "aspect_ratio_ok": None
            },
            "overlay": {
                "frame_width_mm": 50,
                "frame_height_mm": 60,
                "face_height_mm": None,
                "bbox_rel": None
            },
            "metrics": {}
        }

def process_job(job_id: str, saved_file_path: str):
    """
    Background task - job'u işle.
    
    Lifecycle:
    1. Update processing cache for frontend polling
    2. Run analysis
    3. Save to DB (analysis_result, status, requires_ack_ids, can_continue)
    4. Clear processing cache (DB is now source of truth)
    """
    import asyncio
    import asyncpg
    import json
    import re
    
    # Get preview URL from processing cache
    current_status = _processing_jobs.get(job_id, {})
    preview_url = current_status.get("preview_url", None)
        
        # Opsiyonel gecikme (UI için)
    time.sleep(0.5)
    
    # Analiz yap - V2 veya V1
    print(f"🔵 [PIPELINE_START] Job {job_id} - Initial analysis")
    
    if USE_V2_ANALYZER:
        print(f"🔵 [PIPELINE_BRANCH] V2_ANALYZER selected")
        analyze_result = analyze_image_v2(job_id, saved_file_path)
        analyze_result["analysis_source"] = "V2_ANALYZER"
    else:
        print(f"🔵 [PIPELINE_BRANCH] V1_ANALYZER selected (legacy)")
        analyze_result = analyze_image(job_id, saved_file_path)
        analyze_result["analysis_source"] = "V1_ANALYZER"
    
    analyze_result["pipeline_version"] = PIPELINE_VERSION
    
    issue_ids = [i.get("id", "?") for i in analyze_result.get("issues", [])]
    print(f"🔵 [FINAL_STATUS] Job {job_id} - {analyze_result.get('final_status', 'UNKNOWN')} - Issues: {issue_ids}")
    
    # Preview URL'yi koru (orijinal fotoğraf)
    if preview_url:
        analyze_result["preview_url"] = preview_url
        analyze_result["preview_image"] = preview_url
        if "overlay" in analyze_result:
            analyze_result["overlay"]["preview_url"] = preview_url
    
    # Compute DB fields
    final_status = analyze_result.get("final_status", "UNKNOWN")
    db_status = {
        "PASS": JobStatus.PASS,
        "WARN": JobStatus.WARN,
        "FAIL": JobStatus.FAIL
    }.get(final_status, JobStatus.ANALYZED)
    
    requires_ack_ids = [
        issue.get("id") for issue in analyze_result.get("issues", [])
        if issue.get("requires_ack", False)
    ]
    
    can_continue = analyze_result.get("server_can_continue", analyze_result.get("can_continue", False))
    
    # Save to DB - use direct asyncpg connection for sync context
    db_saved = False
    database_url = os.getenv("DATABASE_URL", "")
    
    if database_url:
        asyncpg_url = re.sub(r'^postgres(ql)?://', 'postgresql://', database_url)
        
        async def save_to_db():
            conn = await asyncpg.connect(asyncpg_url)
            try:
                await conn.execute("""
                    UPDATE jobs 
                    SET status = $1,
                        analysis_result = $2::jsonb,
                        normalized_image_path = COALESCE($3, normalized_image_path),
                        requires_ack_ids = $4::jsonb,
                        acknowledged_issue_ids = '[]'::jsonb,
                        can_continue = $5
                    WHERE id = $6::uuid
                """, 
                    db_status,
                    json.dumps(analyze_result, default=str),
                    analyze_result.get("normalized_url"),
                    json.dumps(requires_ack_ids),
                    can_continue,
                    job_id
                )
                return True
            finally:
                await conn.close()
        
        loop = asyncio.new_event_loop()
        try:
            db_saved = loop.run_until_complete(save_to_db())
            print(f"✅ [DB] Job {job_id} saved to database")
        except Exception as e:
            print(f"⚠️ [DB] Failed to save job {job_id}: {e}")
        finally:
            loop.close()
    
    # Clear processing cache - DB is now source of truth
    if db_saved:
        _processing_jobs.pop(job_id, None)
    elif DEV_ALLOW_MEMORY_FALLBACK:
        # Dev mode: keep in processing cache as fallback
        _processing_jobs[job_id] = analyze_result
        print(f"⚠️ [DEV] Keeping job {job_id} in memory fallback")


def process_job_with_bytes(job_id: str, image_bytes: bytes, ext: str, use_supabase: bool):
    """
    Background task - process job with image bytes (for Supabase Storage).
    
    This variant takes image bytes directly instead of a file path,
    avoiding the need to re-download from storage for analysis.
    """
    import asyncio
    import asyncpg
    import json
    import re
    import tempfile
    import traceback
    
    # Get preview URL from processing cache
    current_status = _processing_jobs.get(job_id, {})
    preview_url = current_status.get("preview_url", None)
    
    # Small delay for UI
    time.sleep(0.5)
    
    # Initialize analyze_result to handle errors
    analyze_result = None
    analysis_error = None
    
    # Create a temporary file for analysis (analyzers expect file paths)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name
        
        # Run analysis
        print(f"🔵 [PIPELINE_START] Job {job_id} - Initial analysis (bytes mode)")
        
        if USE_V2_ANALYZER:
            print(f"🔵 [PIPELINE_BRANCH] V2_ANALYZER selected")
            analyze_result = analyze_image_v2(job_id, temp_path)
            analyze_result["analysis_source"] = "V2_ANALYZER"
        else:
            print(f"🔵 [PIPELINE_BRANCH] V1_ANALYZER selected (legacy)")
            analyze_result = analyze_image(job_id, temp_path)
            analyze_result["analysis_source"] = "V1_ANALYZER"
        
        analyze_result["pipeline_version"] = PIPELINE_VERSION
        analyze_result["storage_backend"] = "supabase" if use_supabase else "local"
        
        issue_ids = [i.get("id", "?") for i in analyze_result.get("issues", [])]
        print(f"🔵 [FINAL_STATUS] Job {job_id} - {analyze_result.get('final_status', 'UNKNOWN')} - Issues: {issue_ids}")
        
    except Exception as e:
        # Catch any analysis errors and create a fallback result
        analysis_error = str(e)
        error_tb = traceback.format_exc()
        print(f"❌ [ANALYSIS_ERROR] Job {job_id} - Analysis failed: {analysis_error}")
        print(f"❌ [TRACEBACK] {error_tb}")
        
        # Create error result so job doesn't stay stuck in "processing"
        analyze_result = {
            "final_status": "FAIL",
            "issues": [{
                "id": "ANALYSIS_ERROR",
                "severity": "FAIL",
                "title_tr": "Analiz hatası",
                "message_tr": "Fotoğraf analizi sırasında bir hata oluştu. Lütfen farklı bir fotoğraf deneyin.",
                "requires_ack": False
            }],
            "can_continue": False,
            "server_can_continue": False,
            "error": analysis_error,
            "pipeline_version": PIPELINE_VERSION,
            "storage_backend": "supabase" if use_supabase else "local"
        }
        
    finally:
        # Clean up temp file
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    # Preview URL'yi koru (orijinal fotoğraf)
    if preview_url:
        analyze_result["preview_url"] = preview_url
        analyze_result["preview_image"] = preview_url
        if "overlay" in analyze_result:
            analyze_result["overlay"]["preview_url"] = preview_url
    
    # Compute DB fields
    final_status = analyze_result.get("final_status", "UNKNOWN")
    db_status = {
        "PASS": JobStatus.PASS,
        "WARN": JobStatus.WARN,
        "FAIL": JobStatus.FAIL
    }.get(final_status, JobStatus.ANALYZED)
    
    requires_ack_ids = [
        issue.get("id") for issue in analyze_result.get("issues", [])
        if issue.get("requires_ack", False)
    ]
    
    can_continue = analyze_result.get("server_can_continue", analyze_result.get("can_continue", False))
    
    # Save to DB - use direct asyncpg connection for sync context
    # Import sanitized URL from env_config
    from utils.env_config import get_database_url
    
    db_saved = False
    database_url, _ = get_database_url(required=False)
    
    if database_url:
        # Convert to asyncpg format (postgresql://)
        asyncpg_url = re.sub(r'^postgres(ql)?://', 'postgresql://', database_url)
        
        print(f"🔵 [DB] Attempting to save job {job_id} to database")
        
        async def save_to_db():
            # Disable prepared statement cache for PgBouncer compatibility
            conn = await asyncpg.connect(
                asyncpg_url,
                statement_cache_size=0
            )
            try:
                await conn.execute("""
                    UPDATE jobs 
                    SET status = $1,
                        analysis_result = $2::jsonb,
                        normalized_image_path = COALESCE($3, normalized_image_path),
                        requires_ack_ids = $4::jsonb,
                        acknowledged_issue_ids = '[]'::jsonb,
                        can_continue = $5,
                        updated_at = NOW()
                    WHERE id = $6::uuid
                """, 
                    db_status,
                    json.dumps(analyze_result, default=str),
                    analyze_result.get("normalized_url"),
                    json.dumps(requires_ack_ids),
                    can_continue,
                    job_id
                )
                return True
            finally:
                await conn.close()
        
        loop = asyncio.new_event_loop()
        try:
            db_saved = loop.run_until_complete(save_to_db())
            print(f"✅ [DB] Job {job_id} saved to database")
        except Exception as e:
            print(f"⚠️ [DB] Failed to save job {job_id}: {e}")
            import traceback as tb
            print(f"⚠️ [DB] Traceback: {tb.format_exc()[-500:]}")
        finally:
            loop.close()
    else:
        print(f"⚠️ [DB] No DATABASE_URL configured, cannot save job {job_id}")
    
    # Clear image bytes from processing cache (save memory)
    if job_id in _processing_jobs:
        _processing_jobs[job_id].pop("image_bytes", None)
    
    # Clear processing cache - DB is now source of truth
    if db_saved:
        _processing_jobs.pop(job_id, None)
    elif DEV_ALLOW_MEMORY_FALLBACK:
        # Dev mode: keep in processing cache as fallback
        _processing_jobs[job_id] = analyze_result
        print(f"⚠️ [DEV] Keeping job {job_id} in memory fallback")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/uploads", response_class=HTMLResponse)
async def list_uploads(request: Request):
    """Yüklenen dosyaları listele"""
    uploads_dir = Path("uploads")
    files = []
    
    if uploads_dir.exists():
        for file_path in uploads_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
    
    # Dosyaları tarihe göre sırala (en yeni önce)
    files.sort(key=lambda x: x["modified"], reverse=True)
    
    return templates.TemplateResponse("uploads.html", {
        "request": request,
        "files": files,
        "total_files": len(files),
        "total_size_mb": round(sum(f["size"] for f in files) / (1024 * 1024), 2)
    })

@app.get("/api/uploads", response_class=JSONResponse)
async def list_uploads_api():
    """Yüklenen dosyaları JSON formatında listele"""
    uploads_dir = Path("uploads")
    files = []
    
    if uploads_dir.exists():
        for file_path in uploads_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
    
    files.sort(key=lambda x: x["modified"], reverse=True)
    
    return JSONResponse({
        "total_files": len(files),
        "total_size_mb": round(sum(f["size"] for f in files) / (1024 * 1024), 2),
        "files": files
    })

@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    """Job sayfasını render et - DB-first"""
    # Check processing cache first (brief window)
    if job_id in _processing_jobs:
        return templates.TemplateResponse("job.html", {
            "request": request,
            "job_id": job_id
        })
    
    # DB-first check
    db_job, error = await get_job_safe(job_id)
    
    if error:
        # DB error
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "Veritabanı hatası. Lütfen daha sonra tekrar deneyin."
        }, status_code=503)
    
    if not db_job:
        return templates.TemplateResponse("job_not_found.html", {
            "request": request,
            "job_id": job_id
        }, status_code=404)
    
    return templates.TemplateResponse("job.html", {
        "request": request,
        "job_id": job_id
    })

@app.get("/job/{job_id}/status", response_class=JSONResponse)
async def job_status_endpoint(job_id: str):
    """
    Job status'u JSON olarak döndür.
    
    Production behavior:
    - DB-first reads (no memory precedence)
    - Returns 404 if job not found
    - Returns 503 if DB error (unless DEV_ALLOW_MEMORY_FALLBACK=true)
    """
    # Check if job is currently being processed (brief window)
    if job_id in _processing_jobs:
        return JSONResponse(_processing_jobs[job_id])
    
    # DB-first read
    db_job, error = await get_job_safe(job_id)
    
    if error:
        # DB error - return 503 in production
        print(f"❌ [DB] get_job failed: {error[:100]}")
        return JSONResponse({
            "status": "error",
            "error": "Veritabanı hatası. Lütfen daha sonra tekrar deneyin.",
            "code": "DB_ERROR"
        }, status_code=503)
    
    if not db_job:
        # Job not found
        return JSONResponse({
            "status": "not_found",
            "error": "İş bulunamadı"
        }, status_code=404)
    
    # Format and return
    response = _format_job_response(db_job)
    return JSONResponse(response)


# Request model for process endpoint
class ProcessRequest(BaseModel):
    acknowledged_issue_ids: List[str] = []


@app.post("/process/{job_id}", response_class=JSONResponse)
async def process_photo(job_id: str, request: ProcessRequest):
    """
    Process the photo after user acknowledgement.
    
    This endpoint:
    1. Re-runs analysis with acknowledged_ids to update can_continue
    2. If can_continue is True, processes with PhotoRoom (background removal)
    3. Returns the processed photo URL
    """
    # DB-first job lookup
    db_job, error = await get_job_safe(job_id)
    
    if error:
        return JSONResponse({
            "success": False,
            "error": "Veritabanı hatası"
        }, status_code=503)
    
    if not db_job:
        return JSONResponse({
            "success": False,
            "error": "Job not found"
        }, status_code=404)
    
    # Build current_job from DB
    current_job = db_job.get("analysis_result") or {}
    original_path = db_job.get("original_image_path", "")
    
    # Use /api/preview endpoint for preview URL (handles both Supabase and local)
    current_job["preview_url"] = f"/api/preview/{job_id}"
    
    # Get the original file - check both Supabase and local storage
    saved_file_path = None
    image_bytes = None
    
    # Check if stored in Supabase Storage
    if original_path.startswith("originals/") and is_storage_configured():
        # Download from Supabase
        from utils.supabase_storage import download_bytes_sync
        content, err = download_bytes_sync(original_path)
        if content:
            # Save temporarily for analysis
            ext = os.path.splitext(original_path)[1] or ".jpg"
            saved_file_path = f"uploads/{job_id}{ext}"
            with open(saved_file_path, "wb") as f:
                f.write(content)
            image_bytes = content
    else:
        # Local storage - find the file
        uploads_dir = Path("uploads")
        job_files = list(uploads_dir.glob(f"{job_id}.*"))
        if job_files:
            saved_file_path = str(job_files[0])
    
    if not saved_file_path:
        return JSONResponse({
            "success": False,
            "error": "Orijinal dosya bulunamadı"
        }, status_code=404)
    
    saved_file_path = str(job_files[0])
    acknowledged_ids = request.acknowledged_issue_ids
    
    print(f"🔵 [APP] Processing job {job_id} with acknowledged_ids: {acknowledged_ids}")
    
    # Re-run V2 analysis with acknowledged_ids
    if USE_V2_ANALYZER:
        analyze_result = analyze_image_v2(job_id, saved_file_path, acknowledged_ids=acknowledged_ids)
    else:
        # V1 doesn't support acknowledged_ids
        analyze_result = current_job
    
    # Preserve preview_url
    preview_url = current_job.get("preview_url")
    if preview_url:
        analyze_result["preview_url"] = preview_url
        analyze_result["preview_image"] = preview_url
    
    # Check if we can continue
    can_continue = analyze_result.get("server_can_continue", analyze_result.get("can_continue", False))
    
    if not can_continue:
        # Update and return
        analyze_result["analysis_source"] = "V2_ANALYZER"
        analyze_result["pipeline_version"] = PIPELINE_VERSION
        return JSONResponse({
            "success": False,
            "error": "Cannot continue - unresolved issues",
            "job": analyze_result
        })
    
    # ==========================================================================
    # PHOTOROOM PROCESSING
    # ==========================================================================
    print(f"🔵 [PIPELINE_START] Job {job_id} - PhotoRoom processing")
    print(f"🔵 [PIPELINE_BRANCH] Checking PhotoRoom configuration...")
    
    photoroom_configured = check_photoroom_configured()
    
    if not photoroom_configured:
        print(f"⚠️ [PIPELINE_BRANCH] FALLBACK - PhotoRoom API key not configured")
        analyze_result["analysis_source"] = "FALLBACK"
        analyze_result["pipeline_version"] = PIPELINE_VERSION
        analyze_result["photoroom_error"] = "API key not configured"
        analyze_result["processed"] = False
        analyze_result["status"] = "done"
        return JSONResponse({
            "success": True,
            "job": analyze_result,
            "warning": "PhotoRoom not configured - returning validation only"
        })
    
    print(f"🔵 [PHOTOROOM_CALL_START] Job {job_id}")
    import time as _time
    import tempfile
    photoroom_start = _time.time()
    
    # Determine storage backend
    use_supabase = is_storage_configured()
    original_path = db_job.get("original_image_path", "")
    
    try:
        # Load image - handle both local and Supabase storage
        if use_supabase and original_path.startswith("originals/"):
            # Download from Supabase Storage
            image_bytes, download_error = download_bytes_sync(original_path)
            if download_error or not image_bytes:
                raise ValueError(f"Could not download from storage: {download_error}")
            
            # Decode bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Local file
            image_bgr = cv2.imread(saved_file_path)
        
        if image_bgr is None:
            raise ValueError("Could not load image")
        
        processed_bgr, photoroom_metrics = process_biometric_photo(
            image_bgr,
            job_id=job_id
        )
        
        photoroom_elapsed = _time.time() - photoroom_start
        print(f"🔵 [PHOTOROOM_CALL_END] Job {job_id} - {photoroom_elapsed*1000:.0f}ms - SUCCESS")
        
        # Encode processed image to PNG bytes
        _, png_buffer = cv2.imencode('.png', processed_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        processed_bytes = png_buffer.tobytes()
        
        if use_supabase:
            # Upload processed image to Supabase Storage
            processed_key = get_object_key("processed", job_id, ".png")
            stored_key, storage_error = upload_bytes_sync(processed_key, processed_bytes, "image/png")
            
            if storage_error:
                print(f"⚠️ [STORAGE] Failed to upload processed image: {storage_error}")
                # Fallback to local
                processed_filename = f"{job_id}_processed.png"
                local_path = f"uploads/{processed_filename}"
                with open(local_path, "wb") as f:
                    f.write(processed_bytes)
                processed_url = f"/uploads/{processed_filename}"
                processed_db_path = local_path  # Store local path in DB
            else:
                # Create signed URL for frontend display (24h)
                processed_url, _ = create_signed_url_sync(processed_key, 86400)
                if not processed_url:
                    processed_url = f"/api/download/{job_id}"
                processed_db_path = processed_key  # Store object key in DB
                print(f"✅ [STORAGE] Processed image uploaded: {processed_key}")
        else:
            # Local storage
            processed_filename = f"{job_id}_processed.png"
            local_path = f"uploads/{processed_filename}"
            with open(local_path, "wb") as f:
                f.write(processed_bytes)
            processed_url = f"/uploads/{processed_filename}"
            processed_db_path = local_path  # Store local path in DB
        
        # Update result
        analyze_result["analysis_source"] = "PHOTOROOM"
        analyze_result["pipeline_version"] = PIPELINE_VERSION
        analyze_result["processed"] = True
        analyze_result["processed_url"] = processed_url  # Signed URL or local path for frontend
        analyze_result["processed_db_path"] = processed_db_path  # Object key or local path for DB
        analyze_result["storage_backend"] = "supabase" if use_supabase else "local"
        analyze_result["photoroom_metrics"] = photoroom_metrics
        analyze_result["status"] = "done"
        
        print(f"🔵 [FINAL_STATUS] Job {job_id} - PASS - Issues: {[i['id'] for i in analyze_result.get('issues', [])]}")
        
    except Exception as e:
        photoroom_elapsed = _time.time() - photoroom_start
        error_str = str(e)
        print(f"❌ [PHOTOROOM_CALL_END] Job {job_id} - {photoroom_elapsed*1000:.0f}ms - FAILED: {error_str}")
        
        # Check if it's a rate limit error
        is_rate_limit = "429" in error_str or "throttled" in error_str.lower()
        
        analyze_result["analysis_source"] = "FALLBACK"
        analyze_result["pipeline_version"] = PIPELINE_VERSION
        analyze_result["photoroom_error"] = error_str
        analyze_result["processed"] = False
        analyze_result["status"] = "done"
        
        if is_rate_limit:
            return JSONResponse({
                "success": False,
                "error": "PhotoRoom API rate limited. Please try again later.",
                "error_code": "RATE_LIMITED",
                "job": analyze_result
            })
        
        return JSONResponse({
            "success": False,
            "error": f"PhotoRoom processing failed: {error_str}",
            "job": analyze_result
        })
    
    # Update DB with processed image path (store object key, not signed URL)
    updated_job, db_error = await update_job_safe(
        job_id, 
        processed_image_path=analyze_result.get("processed_db_path")
    )
    
    if db_error:
        print(f"⚠️ [DB] Failed to update processed path: {db_error}")
    else:
        print(f"✅ [DB] Processed image path saved for job {job_id}")
    
    return JSONResponse({
        "success": True,
        "job": analyze_result
    })


@app.post("/upload", response_class=JSONResponse)
async def upload_file(request: Request, background_tasks: BackgroundTasks, photo: UploadFile = File(...)):
    # Dosya uzantısı kontrolü
    if not is_allowed_file(photo.filename):
        return JSONResponse({
            "success": False,
            "error": f"Sadece .jpg, .jpeg, .png ve .webp dosyaları kabul edilir. Yüklenen dosya: {photo.filename}",
            "job_id": None
        }, status_code=400)
    
    # Dosyayı oku
    try:
        content = await photo.read()
        
        # Boş dosya kontrolü
        if len(content) == 0:
            return JSONResponse({
                "success": False,
                "error": "Boş dosya yüklenemez.",
                "job_id": None
            }, status_code=400)
        
        # Dosya boyutu kontrolü (10MB limit for storage)
        MAX_STORAGE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(content) > MAX_STORAGE_SIZE:
            return JSONResponse({
                "success": False,
                "error": "Dosya çok büyük, maksimum 10MB",
                "job_id": None
            }, status_code=413)
        
        # Validate image magic bytes
        is_valid, detected_type, validation_error = validate_image_bytes(content)
        if not is_valid:
            return JSONResponse({
                "success": False,
                "error": f"Geçersiz dosya formatı: {validation_error}",
                "job_id": None
            }, status_code=400)
        
        # job_id üret
        job_id = str(uuid4())
        
        # Dosya uzantısını al
        ext = get_file_extension(photo.filename)
        content_type = get_content_type(ext)
        
        # Determine storage backend
        use_supabase = is_storage_configured()
        
        if use_supabase:
            # Upload to Supabase Storage
            object_key = get_object_key("originals", job_id, ext)
            stored_key, storage_error = await upload_bytes(object_key, content, content_type)
            
            if storage_error:
                print(f"❌ [STORAGE] Upload failed: {storage_error}")
                return JSONResponse({
                    "success": False,
                    "error": "Dosya yüklenirken bir sorun oluştu. Lütfen tekrar deneyin.",
                    "job_id": None
                }, status_code=503)
            
            # Store object key in DB (not local path)
            original_image_path = object_key
            print(f"✅ [UPLOAD] Stored in Supabase: {object_key}")
        else:
            # Fallback: Local filesystem storage
            saved_file_path = f"uploads/{job_id}{ext}"
            with open(saved_file_path, "wb") as buffer:
                buffer.write(content)
            
            original_image_path = saved_file_path
            print(f"✅ [UPLOAD] Stored locally: {saved_file_path}")
        
        # Always use /api/preview endpoint - it handles both Supabase and local storage
        # and generates fresh signed URLs when needed
        preview_url = f"/api/preview/{job_id}"
        
        # Job'u veritabanında oluştur
        db_job, db_error = await create_job_safe(
            job_id=job_id,
            original_image_path=original_image_path,
            status=JobStatus.PROCESSING
        )
        
        if db_error and not DEV_ALLOW_MEMORY_FALLBACK:
            # Production: DB failure is fatal for upload
            print(f"❌ [DB] create_job failed: {db_error}")
            return JSONResponse({
                "success": False,
                "error": "Veritabanı hatası. Lütfen daha sonra tekrar deneyin.",
                "job_id": None
            }, status_code=503)
        
        # Store in processing cache for the brief analysis window
        _processing_jobs[job_id] = {
            "status": "processing",
            "result": None,
            "reasons": [],
            "preview_url": preview_url,
            "image_bytes": content,  # Keep bytes for background analysis
            "use_supabase": use_supabase
        }
        
        # Background task ile process_job başlat
        # Pass content bytes for analysis (avoids re-download)
        background_tasks.add_task(process_job_with_bytes, job_id, content, ext, use_supabase)
        
        # Başarılı yükleme - JSON response
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "preview_url": preview_url,
            "storage": "supabase" if use_supabase else "local",
            "message": "Fotoğraf başarıyla yüklendi"
        })
    except Exception as e:
        print(f"❌ [UPLOAD] Exception: {e}")
        return JSONResponse({
            "success": False,
            "error": f"Dosya kaydedilirken bir sorun oluştu: {str(e)}",
            "job_id": None
        }, status_code=500)


# ============================================================================
# Payment Endpoints
# ============================================================================

class ShippingInfo(BaseModel):
    first_name: str
    last_name: str
    address: str
    city: str
    postal_code: str
    phone: str
    email: str


class CheckoutRequest(BaseModel):
    job_id: str
    package_type: str  # "digital" or "digital_print"
    shipping: Optional[ShippingInfo] = None


@app.post("/api/checkout", response_class=JSONResponse)
async def create_checkout(request: Request, checkout_req: CheckoutRequest):
    """
    Create a Stripe Checkout session for payment.
    Returns 501 if payments are disabled.
    """
    # Guard: Check if payments are enabled
    if not is_payments_enabled():
        return JSONResponse({
            "success": False,
            "error": "Ödeme sistemi şu anda aktif değil",
            "code": "PAYMENTS_DISABLED"
        }, status_code=501)
    
    job_id = checkout_req.job_id
    package_type = checkout_req.package_type
    
    # DB-first job validation
    db_job, error = await get_job_safe(job_id)
    
    if error:
        return JSONResponse({
            "success": False,
            "error": "Veritabanı hatası"
        }, status_code=503)
    
    if not db_job:
        return JSONResponse({
            "success": False,
            "error": "İş bulunamadı"
        }, status_code=404)
    
    # Validate package type
    if package_type not in ["digital", "digital_print"]:
        return JSONResponse({
            "success": False,
            "error": "Geçersiz paket tipi"
        }, status_code=400)
    
    # Require shipping info for digital_print
    if package_type == "digital_print" and not checkout_req.shipping:
        return JSONResponse({
            "success": False,
            "error": "Baskı siparişi için teslimat bilgileri gerekli"
        }, status_code=400)
    
    # Store shipping info (email in jobs table, print_orders table later)
    if checkout_req.shipping:
        # Update user email and payment state in DB
        _, db_error = await update_job_safe(
            job_id,
            user_email=checkout_req.shipping.email,
            payment_state=PaymentState.PAYMENT_PENDING
        )
        if db_error:
            print(f"⚠️ [DB] Failed to update email: {db_error}")
        else:
            print(f"[CHECKOUT] Shipping info saved for job {job_id}")
    
    # Build URLs
    base_url = str(request.base_url).rstrip("/")
    success_url = f"{base_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{base_url}/payment/cancel?job_id={job_id}"
    
    try:
        result = create_checkout_session(
            job_id=job_id,
            package_type=package_type,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        
        return JSONResponse({
            "success": True,
            "checkout_url": result["checkout_url"],
            "session_id": result["session_id"],
        })
        
    except ValueError as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
    except Exception as e:
        print(f"❌ [CHECKOUT] Error: {e}")
        return JSONResponse({
            "success": False,
            "error": "Ödeme oturumu oluşturulamadı"
        }, status_code=500)


@app.get("/payment/success", response_class=HTMLResponse)
async def payment_success(request: Request, session_id: str):
    """
    Handle successful payment redirect from Stripe.
    Note: This page should NOT trust the session_id alone - webhook verification is required.
    """
    # Try to get job_id from Stripe session
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        job_id = session.metadata.get("job_id")
    except:
        job_id = None
    
    return templates.TemplateResponse("payment_success.html", {
        "request": request,
        "session_id": session_id,
        "job_id": job_id,
    })


@app.get("/payment/cancel", response_class=HTMLResponse)
async def payment_cancel(request: Request, job_id: str = None):
    """Handle cancelled payment redirect from Stripe."""
    return templates.TemplateResponse("payment_cancel.html", {
        "request": request,
        "job_id": job_id,
    })


@app.post("/api/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhooks for payment verification.
    
    This endpoint verifies the webhook signature and processes
    checkout.session.completed events to mark orders as paid.
    
    Returns 501 if payments are disabled.
    """
    # Guard: Check if payments are enabled
    if not is_payments_enabled():
        return JSONResponse({
            "error": "Payments disabled"
        }, status_code=501)
    
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    
    # Verify signature
    if not verify_webhook_signature(payload, sig_header):
        print("❌ [WEBHOOK] Invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    try:
        # Import stripe only when needed
        import stripe as _stripe
        event = _stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        print(f"❌ [WEBHOOK] Error constructing event: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        success = handle_checkout_completed(session)
        
        if success:
            print(f"✅ [WEBHOOK] Checkout completed for job: {session.get('metadata', {}).get('job_id')}")
        else:
            print(f"⚠️ [WEBHOOK] Failed to process checkout completion")
    
    return JSONResponse({"received": True})


@app.get("/api/payment/status/{job_id}", response_class=JSONResponse)
async def get_payment_status(job_id: str):
    """
    Get payment status for a job.
    
    If payments are disabled:
    - Returns enabled=false
    - can_download=true (free download during beta)
    """
    # Check if payments are enabled
    payments_enabled = is_payments_enabled()
    
    if not payments_enabled:
        # Payments disabled - allow free download
        return JSONResponse({
            "enabled": False,
            "paid": True,  # Treat as paid when payments disabled
            "state": "FREE",
            "can_download": True,
            "download_url": generate_signed_url(job_id),
            "message": "Beta döneminde indirme ücretsiz"
        })
    
    order = get_order(job_id)
    
    if not order:
        # Check if job exists in DB
        db_job, _ = await get_job_safe(job_id)
        if db_job:
            # Job exists but no payment attempted
            return JSONResponse({
                "enabled": True,
                "paid": False,
                "state": "ANALYZED",
                "can_download": False,
            })
        return JSONResponse({
            "error": "İş bulunamadı"
        }, status_code=404)
    
    paid = is_paid(job_id)
    
    return JSONResponse({
        "enabled": True,
        "paid": paid,
        "state": order.get("state"),
        "package_type": order.get("package_type"),
        "can_download": paid,
        "download_url": generate_signed_url(job_id) if paid else None,
    })


@app.get("/api/download/{job_id}")
async def secure_download(job_id: str, expires: int = None, sig: str = None):
    """
    Secure download endpoint for processed photos.
    
    Two modes:
    1. Signed URL mode (with expires + sig): Verify signature
    2. Direct mode (no sig): Verify payment status server-side
    
    If using Supabase Storage, redirects to a signed URL.
    If using local storage, serves the file directly.
    """
    # If signed URL parameters provided, verify them
    if expires is not None and sig is not None:
        if not verify_signed_url(job_id, expires, sig):
            raise HTTPException(status_code=403, detail="Geçersiz veya süresi dolmuş link")
    else:
        # No signed URL - verify payment status (skip if payments disabled)
        if is_payments_enabled() and not is_paid(job_id):
            raise HTTPException(status_code=403, detail="İndirme için ödeme gerekli")
    
    # Get job from DB to check storage backend
    db_job, db_error = await get_job_safe(job_id)
    
    if db_error or not db_job:
        raise HTTPException(status_code=404, detail="İş bulunamadı")
    
    processed_path = db_job.get("processed_image_path", "")
    
    # Check if stored in Supabase Storage (path starts with "processed/")
    if processed_path.startswith("processed/") and is_storage_configured():
        # Redirect to Supabase signed URL
        signed_url, error = await create_signed_url(processed_path, 3600)  # 1 hour
        
        if error or not signed_url:
            raise HTTPException(status_code=500, detail="İndirme linki oluşturulamadı")
        
        # Return redirect to signed URL with download headers
        return RedirectResponse(
            url=signed_url,
            status_code=302,
            headers={
                "Content-Disposition": f'attachment; filename="biyometrik_foto_{job_id[:8]}.png"'
            }
        )
    else:
        # Local file fallback
        local_path = Path(f"uploads/{job_id}_processed.png")
        
        if not local_path.exists():
            raise HTTPException(status_code=404, detail="Dosya bulunamadı")
        
        return FileResponse(
            path=str(local_path),
            filename=f"biyometrik_foto_{job_id[:8]}.png",
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="biyometrik_foto_{job_id[:8]}.png"'
            }
        )


class EmailLinkRequest(BaseModel):
    job_id: str
    email: str


@app.post("/api/email-link", response_class=JSONResponse)
async def send_email_link(request: Request, email_req: EmailLinkRequest):
    """
    Send download link to user's email (optional, after payment).
    """
    job_id = email_req.job_id
    email = email_req.email
    
    # Verify payment
    if not is_paid(job_id):
        return JSONResponse({
            "success": False,
            "error": "E-posta linki göndermek için ödeme gerekli"
        }, status_code=403)
    
    # Generate signed URL (24 hour expiry)
    download_url = generate_email_link(job_id, expires_in_hours=24)
    
    # Get base URL
    base_url = str(request.base_url).rstrip("/")
    
    # Send email
    result = send_download_link(
        job_id=job_id,
        email=email,
        download_url=download_url,
        base_url=base_url,
    )
    
    if result["success"]:
        # Update order state
        update_order_state(job_id, "DELIVERED", email_sent_to=email)
    
    return JSONResponse(result)


@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """
    Get preview URL for a job's original image.
    
    If using Supabase Storage, returns a redirect to a signed URL.
    If using local storage, redirects to the local file.
    """
    # Get job from DB
    db_job, db_error = await get_job_safe(job_id)
    
    if db_error or not db_job:
        raise HTTPException(status_code=404, detail="İş bulunamadı")
    
    original_path = db_job.get("original_image_path", "")
    
    # Check if stored in Supabase Storage (path starts with "originals/")
    if original_path.startswith("originals/") and is_storage_configured():
        # Redirect to Supabase signed URL
        signed_url, error = await create_signed_url(original_path, 3600)  # 1 hour
        
        if error or not signed_url:
            raise HTTPException(status_code=500, detail="Önizleme linki oluşturulamadı")
        
        return RedirectResponse(url=signed_url, status_code=302)
    else:
        # Local file - redirect to static file
        ext = os.path.splitext(original_path)[1] or ".jpg"
        local_url = f"/uploads/{job_id}{ext}"
        return RedirectResponse(url=local_url, status_code=302)


@app.get("/api/health/storage", response_class=JSONResponse)
async def storage_health():
    """Check Supabase Storage health."""
    configured = is_storage_configured()
    
    if not configured:
        return JSONResponse({
            "ok": False,
            "configured": False,
            "message": "Supabase Storage not configured"
        })
    
    # Try to list bucket (basic health check)
    try:
        from utils.supabase_storage import get_storage_client, _get_config
        client = get_storage_client()
        config = _get_config()
        bucket = config["bucket"]
        
        # Try to list root of bucket (will fail if bucket doesn't exist or no access)
        # Note: No 'limit' param - not supported in supabase-py storage3
        result = client.storage.from_(bucket).list(path="")
        
        return JSONResponse({
            "ok": True,
            "configured": True,
            "bucket": bucket,
            "message": "Supabase Storage healthy"
        })
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "configured": True,
            "error": str(e)[:100],
            "message": "Supabase Storage error"
        })


@app.get("/api/config-check", response_class=JSONResponse)
async def config_check():
    """
    Configuration check endpoint for debugging.
    
    Returns non-secret configuration summary:
    - db_configured: bool
    - db_host: hostname (no credentials)
    - storage_configured: bool
    - payments_enabled: bool
    - photoroom_configured: bool
    - warnings: list of config warnings
    """
    summary = get_config_summary()
    
    return JSONResponse({
        "db": {
            "configured": summary["db_configured"],
            "host": summary["db_host"],
            "port": summary["db_port"],
            "warnings": summary["db_warnings"]
        },
        "storage": {
            "configured": summary["storage_configured"],
            "bucket": summary["storage_bucket"],
            "project_ref": summary["supabase_project_ref"],
            "warnings": summary["supabase_warnings"],
            "missing_vars": summary.get("storage_missing_vars", [])
        },
        "payments": {
            "enabled": summary["payments_enabled"]
        },
        "photoroom": {
            "configured": summary["photoroom_configured"]
        }
    })


@app.get("/api/health/analyzer", response_class=JSONResponse)
async def analyzer_health():
    """
    Check if the face analyzer is working.
    
    Tests:
    - MediaPipe model loading
    - OpenCV availability
    - Basic image processing
    """
    import traceback
    
    results = {
        "opencv": {"ok": False, "version": None},
        "mediapipe": {"ok": False, "version": None},
        "analyzer_v2": {"ok": False, "error": None},
    }
    
    # Check OpenCV
    try:
        import cv2
        results["opencv"]["ok"] = True
        results["opencv"]["version"] = cv2.__version__
    except Exception as e:
        results["opencv"]["error"] = str(e)[:100]
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        results["mediapipe"]["ok"] = True
        results["mediapipe"]["version"] = mp.__version__
    except Exception as e:
        results["mediapipe"]["error"] = str(e)[:100]
    
    # Check Analyzer V2
    try:
        from utils.analyze_v2 import analyze_image_v2
        # Create a small test image
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:] = (255, 255, 255)  # White background
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, test_img)
            temp_path = tmp.name
        
        # Try to analyze (should return FACE_NOT_FOUND for blank image)
        try:
            result = analyze_image_v2("test", temp_path)
            results["analyzer_v2"]["ok"] = True
            results["analyzer_v2"]["final_status"] = result.get("final_status")
            results["analyzer_v2"]["issues"] = [i.get("id") for i in result.get("issues", [])]
        finally:
            import os
            os.unlink(temp_path)
    except Exception as e:
        results["analyzer_v2"]["error"] = str(e)[:200]
        results["analyzer_v2"]["traceback"] = traceback.format_exc()[-500:]
    
    all_ok = all([
        results["opencv"]["ok"],
        results["mediapipe"]["ok"],
        results["analyzer_v2"]["ok"]
    ])
    
    return JSONResponse({
        "ok": all_ok,
        "results": results
    })


@app.get("/api/config/stripe", response_class=JSONResponse)
async def get_stripe_config():
    """
    Get payment configuration for frontend.
    Returns enabled status and prices even when payments are disabled.
    """
    config = get_payments_config()
    return JSONResponse({
        "enabled": config["enabled"],
        "publishable_key": config.get("publishable_key", ""),
        "message": config.get("message", ""),
        "prices": {
            "digital": config["digital_price"],
            "digital_print": config["print_price"],
        }
    })
