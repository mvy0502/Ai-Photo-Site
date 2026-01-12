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
from utils.job_queue import get_job_queue, init_job_queue, JobState
from utils.rate_limiter import get_rate_limiter, get_status_cache
from utils.db_pool import get_db_pool, close_db_pool

# PayTR Payment Integration
from utils.paytr import (
    is_paytr_configured, get_paytr_config, create_iframe_token,
    verify_webhook_hash, parse_merchant_oid, get_client_ip,
    get_price_display as get_paytr_price_display, get_iframe_url,
    DIGITAL_PRICE_KURUS as PAYTR_DIGITAL_PRICE,
    PRINT_PRICE_KURUS as PAYTR_PRINT_PRICE
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
    """Initialize database connection, job queue, and validate configuration on startup."""
    import time as startup_time
    startup_start = startup_time.time()
    
    # Run config validation first
    startup_validation()
    
    # Validate DATABASE_URL format
    _validate_database_url()
    
    # Initialize async database (for API reads)
    success, message = await db_manager.initialize()
    if success:
        print(f"✅ Database (async): {message}")
    else:
        print(f"⚠️ Database (async): {message}")
        print("   App will continue without database. Some features may be unavailable.")
    
    # Initialize sync DB pool (for background worker writes)
    print("📦 [STARTUP] Initializing DB connection pool...")
    pool = get_db_pool()
    if pool and pool.is_initialized():
        print("✅ [STARTUP] DB pool ready")
    else:
        print("⚠️ [STARTUP] DB pool not available (will use direct connections)")
    
    # Pre-warm MediaPipe analyzer (expensive one-time init)
    print("🔧 [STARTUP] Pre-warming MediaPipe FaceLandmarker...")
    from utils.analyzer_v2 import get_analyzer, get_analyzer_init_time_ms
    analyzer = get_analyzer()
    if analyzer:
        print(f"✅ [STARTUP] MediaPipe ready (init took {get_analyzer_init_time_ms()}ms)")
    else:
        print("⚠️ [STARTUP] MediaPipe not available - analysis will fail")
    
    # Initialize job queue with worker
    print("📦 [STARTUP] Initializing job queue...")
    init_job_queue(process_job_worker)
    print("✅ [STARTUP] Job queue ready")
    
    startup_ms = int((startup_time.time() - startup_start) * 1000)
    print(f"🚀 [STARTUP] Complete in {startup_ms}ms")


def _validate_database_url():
    """Validate DATABASE_URL and log warnings about pooler vs direct connection."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    
    if not database_url:
        print("⚠️ [DB_URL] DATABASE_URL not set")
        return
    
    # Parse host from URL
    try:
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        host = parsed.hostname or ""
        port = parsed.port or 5432
        
        is_pooler = "pooler" in host.lower()
        is_supabase = "supabase" in host.lower()
        
        print(f"📊 [DB_URL] Host: {host[:30]}..., Port: {port}")
        
        if is_pooler:
            print("⚠️ [DB_URL] Using Supabase POOLER connection (aws-xxx.pooler.supabase.com)")
            print("   - Session mode (port 5432): OK for most operations")
            print("   - Transaction mode (port 6543): May have prepared statement issues")
            if port == 6543:
                print("   🔶 RECOMMENDATION: Consider using port 5432 (session mode) for better compatibility")
        elif is_supabase:
            print("✅ [DB_URL] Using Supabase DIRECT connection")
        else:
            print(f"📊 [DB_URL] Using external database: {host[:20]}...")
            
    except Exception as e:
        print(f"⚠️ [DB_URL] Could not parse DATABASE_URL: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections and stop job queue on shutdown."""
    # Stop job queue
    try:
        queue = get_job_queue()
        queue.stop_worker()
        print("🛑 Job queue stopped.")
    except:
        pass
    
    # Close sync DB pool
    try:
        close_db_pool()
        print("🔌 DB pool closed.")
    except:
        pass
    
    # Close async database
    await db_manager.close()
    print("🔌 Database connections closed.")

@app.get("/api/health", response_class=JSONResponse)
async def health():
    """
    Basic health check endpoint for load balancers and uptime monitoring.
    Returns minimal info without exposing internals.
    
    CRITICAL: Must return immediately (<100ms).
    Does NOT touch DB, Storage, or external APIs.
    """
    return JSONResponse({
        "status": "ok",
        "service": "biyometrikfoto-api",
        "version": PIPELINE_VERSION
    })


@app.get("/api/queue/stats", response_class=JSONResponse)
async def queue_stats():
    """
    Get queue statistics for monitoring.
    """
    queue = get_job_queue()
    stats = queue.get_queue_stats()
    return JSONResponse(stats)


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


def process_job_with_path(job_id: str, path_or_key: str, ext: str):
    """
    Background task - process job by downloading from storage or reading local file.
    
    IMPORTANT: Job completion is NOT dependent on DB writes.
    1. Download image
    2. Run analysis
    3. Mark DONE/FAILED in memory IMMEDIATELY
    4. Attempt DB write (non-blocking - retry on failure)
    
    Args:
        job_id: Job UUID
        path_or_key: Either Supabase object key (e.g., "originals/xxx.jpg") 
                     or local file path (e.g., "uploads/xxx.jpg")
        ext: File extension (e.g., ".jpg")
    """
    import json
    import re
    import tempfile
    import traceback
    import psycopg2
    import time as timing
    from utils.env_config import get_database_url
    from utils.supabase_storage import download_bytes_sync, is_storage_configured
    
    # Timing metrics
    total_start = timing.time()
    download_ms = 0
    analyze_ms = 0
    db_ms = 0
    
    print(f"🟢 [BG_START] Job {job_id} - path: {path_or_key}, ext: {ext}")
    
    # Preview URL for this job
    preview_url = f"/api/preview/{job_id}"
    
    # Initialize
    analyze_result = None
    db_saved = False
    db_error = None
    job_queue = get_job_queue()
    
    try:
        # Step 1: Get image bytes (from storage or local file)
        download_start = timing.time()
        is_storage_path = path_or_key.startswith("originals/") or path_or_key.startswith("processed/")
        
        if is_storage_path and is_storage_configured():
            # Download from Supabase Storage
            print(f"🟣 [BG_DOWNLOAD] Downloading {path_or_key} from Supabase...")
            image_bytes, download_error = download_bytes_sync(path_or_key)
            
            if download_error or not image_bytes:
                raise Exception(f"Failed to download from storage: {download_error}")
        else:
            # Read from local file
            print(f"🟣 [BG_READ] Reading from local file: {path_or_key}")
            if not os.path.exists(path_or_key):
                raise Exception(f"Local file not found: {path_or_key}")
            
            with open(path_or_key, "rb") as f:
                image_bytes = f.read()
        
        download_ms = int((timing.time() - download_start) * 1000)
        print(f"🟣 [BG_DOWNLOADED] Got {len(image_bytes)} bytes in {download_ms}ms")
        
        # Step 2: Create temp file and run analysis
        analyze_start = timing.time()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(image_bytes)
                temp_path = tmp.name
            
            # Free memory - don't hold image bytes during analysis
            image_bytes = None
            
            print(f"🔵 [BG_ANALYZE] Running analyzer on {temp_path}")
            
            if USE_V2_ANALYZER:
                analyze_result = analyze_image_v2(job_id, temp_path)
                analyze_result["analysis_source"] = "V2_ANALYZER"
            else:
                analyze_result = analyze_image(job_id, temp_path)
                analyze_result["analysis_source"] = "V1_ANALYZER"
            
            analyze_result["pipeline_version"] = PIPELINE_VERSION
            analyze_result["storage_backend"] = "supabase"
            
            issue_ids = [i.get("id", "?") for i in analyze_result.get("issues", [])]
            analyze_ms = int((timing.time() - analyze_start) * 1000)
            print(f"🔵 [BG_RESULT] {analyze_result.get('final_status', 'UNKNOWN')} - Issues: {issue_ids} ({analyze_ms}ms)")
            
        finally:
            # Always delete temp file
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # Add preview URL
        analyze_result["preview_url"] = preview_url
        analyze_result["preview_image"] = preview_url
        
    except Exception as e:
        # Create error result
        error_msg = str(e)
        error_tb = traceback.format_exc()
        print(f"🔴 [BG_FAIL] Job {job_id} failed: {error_msg}")
        print(f"🔴 [BG_TRACEBACK] {error_tb[-500:]}")
        
        analyze_result = {
            "final_status": "FAIL",
            "issues": [{
                "id": "ANALYSIS_ERROR",
                "severity": "FAIL",
                "title_tr": "Analiz hatası",
                "message_tr": "Fotoğraf analizi sırasında bir hata oluştu. Lütfen tekrar deneyin.",
                "requires_ack": False
            }],
            "can_continue": False,
            "server_can_continue": False,
            "error": error_msg,
            "pipeline_version": PIPELINE_VERSION,
            "storage_backend": "supabase",
            "preview_url": preview_url,
            "preview_image": preview_url
        }
    
    # =================================================================
    # CRITICAL: Mark job DONE/FAILED in memory BEFORE DB write
    # Job completion should NOT depend on DB availability
    # =================================================================
    final_status = analyze_result.get("final_status", "UNKNOWN")
    
    # Store result in processing cache for status endpoint
    _processing_jobs[job_id] = {
        "status": "done",
        "final_status": final_status,
        "result": analyze_result,
        "preview_url": preview_url,
        **analyze_result  # Spread all result fields
    }
    
    # Mark in queue (if available)
    if job_queue:
        if final_status == "FAIL":
            job_queue.mark_failed(job_id, analyze_result.get("error", "Unknown error"), analyze_result)
        else:
            job_queue.mark_done(job_id, analyze_result)
    
    total_before_db = int((timing.time() - total_start) * 1000)
    print(f"✅ [BG_MEMORY] Job {job_id} marked {final_status} in memory ({total_before_db}ms)")
    
    # =================================================================
    # Step 3: Attempt DB write (non-blocking - job already DONE)
    # =================================================================
    db_start = timing.time()
    
    db_status = {
        "PASS": JobStatus.PASS,
        "WARN": JobStatus.WARN,
        "FAIL": JobStatus.FAIL
    }.get(final_status, JobStatus.FAIL)
    
    requires_ack_ids = [
        issue.get("id") for issue in analyze_result.get("issues", [])
        if issue.get("requires_ack", False)
    ]
    can_continue = analyze_result.get("server_can_continue", analyze_result.get("can_continue", False))
    
    print(f"🔵 [BG_DB] Saving job {job_id} to database (non-blocking)...")
    
    # Try to use connection pool first
    pool = get_db_pool()
    
    if pool and pool.is_initialized():
        try:
            analysis_json = json.dumps(analyze_result, default=str)
            requires_ack_json = json.dumps(requires_ack_ids)
            
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE jobs 
                        SET status = %s,
                            analysis_result = %s::jsonb,
                            requires_ack_ids = %s::jsonb,
                            acknowledged_issue_ids = '[]'::jsonb,
                            can_continue = %s,
                            updated_at = NOW()
                        WHERE id = %s::uuid
                    """, (
                        db_status,
                        analysis_json,
                        requires_ack_json,
                        can_continue,
                        job_id
                    ))
                    conn.commit()
                    db_saved = True
                    
        except Exception as e:
            db_error = str(e)[:200]
            print(f"⚠️ [BG_DB_FAIL] Pool error (job still DONE): {db_error}")
            
    else:
        # Fallback: direct connection
        database_url, _ = get_database_url(required=False)
        
        if database_url:
            psycopg_url = re.sub(r'^postgres://', 'postgresql://', database_url)
            conn = None
            cur = None
            
            try:
                conn = psycopg2.connect(
                    psycopg_url, 
                    connect_timeout=10,
                    options='-c statement_timeout=30000'
                )
                conn.autocommit = False
                cur = conn.cursor()
                
                analysis_json = json.dumps(analyze_result, default=str)
                requires_ack_json = json.dumps(requires_ack_ids)
                
                cur.execute("""
                    UPDATE jobs 
                    SET status = %s,
                        analysis_result = %s::jsonb,
                        requires_ack_ids = %s::jsonb,
                        acknowledged_issue_ids = '[]'::jsonb,
                        can_continue = %s,
                        updated_at = NOW()
                    WHERE id = %s::uuid
                """, (
                    db_status,
                    analysis_json,
                    requires_ack_json,
                    can_continue,
                    job_id
                ))
                
                conn.commit()
                db_saved = True
                
            except Exception as e:
                db_error = str(e)[:200]
                print(f"⚠️ [BG_DB_FAIL] Direct conn error (job still DONE): {db_error}")
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                        
            finally:
                if cur:
                    try:
                        cur.close()
                    except:
                        pass
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
        else:
            db_error = "No DATABASE_URL configured"
            print(f"⚠️ [BG_DB_FAIL] {db_error} (job still DONE)")
    
    db_ms = int((timing.time() - db_start) * 1000)
    total_ms = int((timing.time() - total_start) * 1000)
    
    # Update job queue with timing and DB status
    if job_queue:
        job_queue.update_timing(job_id, download_ms, analyze_ms, db_ms, total_ms)
        job_queue.update_db_status(job_id, db_saved, db_error)
    
    # Add db_saved to processing cache for status endpoint
    if job_id in _processing_jobs:
        _processing_jobs[job_id]["db_saved"] = db_saved
        _processing_jobs[job_id]["db_error"] = db_error
        _processing_jobs[job_id]["timing"] = {
            "download_ms": download_ms,
            "analyze_ms": analyze_ms,
            "db_ms": db_ms,
            "total_ms": total_ms
        }
    
    # Log final timing summary
    print(f"📊 [BG_TIMING] Job {job_id}: download={download_ms}ms, analyze={analyze_ms}ms, db={db_ms}ms, total={total_ms}ms")
    print(f"✅ [BG_COMPLETE] Job {job_id} - status={final_status}, db_saved={db_saved}")
    
    # Invalidate status cache after processing
    try:
        status_cache = get_status_cache()
        status_cache.invalidate(job_id)
    except:
        pass
    
    return db_saved


def process_job_worker(job_id: str, object_key: str, ext: str) -> bool:
    """
    Worker function called by job queue.
    Wraps process_job_with_path and returns success boolean.
    
    This function runs in the queue worker thread (not the main event loop).
    """
    print(f"⚙️ [WORKER] Starting job {job_id}")
    
    try:
        success = process_job_with_path(job_id, object_key, ext)
        print(f"✅ [WORKER] Job {job_id} completed - success: {success}")
        return success
    except Exception as e:
        print(f"🔴 [WORKER] Job {job_id} exception: {e}")
        import traceback
        traceback.print_exc()
        return False


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
async def job_status_endpoint(job_id: str, request: Request):
    """
    Job status'u JSON olarak döndür.
    
    Production behavior:
    - Rate limited (1 request/second per job_id)
    - Cached responses (1s TTL)
    - Checks queue status first, then DB
    - Returns 429 if polling too fast
    """
    # Rate limiting - prevent aggressive polling
    rate_limiter = get_rate_limiter()
    allowed, retry_after = rate_limiter.check(f"status:{job_id}")
    
    if not allowed:
        return JSONResponse({
            "status": "rate_limited",
            "error": "Too many requests. Please slow down.",
            "retry_after": retry_after
        }, status_code=429, headers={"Retry-After": str(int(retry_after + 1))})
    
    # Check cache first
    status_cache = get_status_cache()
    cached = status_cache.get(job_id)
    if cached:
        return JSONResponse(cached)
    
    # Check queue status first (most recent state)
    job_queue = get_job_queue()
    queue_status = job_queue.get_status(job_id)
    
    if queue_status:
        state = queue_status.get("state")
        
        if state == JobState.QUEUED.value:
            response = {
                "status": "queued",
                "queue_position": queue_status.get("queue_position", 0),
                "message": "Fotoğrafınız sırada bekliyor..."
            }
            status_cache.set(job_id, response)
            return JSONResponse(response)
        
        elif state == JobState.PROCESSING.value:
            response = {
                "status": "processing",
                "message": "Fotoğrafınız analiz ediliyor..."
            }
            status_cache.set(job_id, response)
            return JSONResponse(response)
        
        elif state in (JobState.DONE.value, JobState.FAILED.value):
            # Job completed in queue - check memory first for full result
            if job_id in _processing_jobs:
                response = _processing_jobs[job_id].copy()
                # Include queue-level db_saved status
                response["db_saved"] = queue_status.get("db_saved", False)
                response["db_error"] = queue_status.get("db_error")
                response["timing"] = queue_status.get("timing", {})
                status_cache.set(job_id, response)
                return JSONResponse(response)
    
    # Check if job is currently being processed (brief window)
    if job_id in _processing_jobs:
        response = _processing_jobs[job_id].copy()
        # Ensure db_saved is included
        response.setdefault("db_saved", True)  # Default to true if not set
        response.setdefault("db_error", None)
        status_cache.set(job_id, response)
        return JSONResponse(response)
    
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
    
    # Cache final states longer
    final_status = response.get("status", "")
    if final_status in ["PASS", "WARN", "FAIL", "done"]:
        status_cache.set(job_id, response)
    
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
        # Download from Supabase (use async version - we're in async endpoint)
        content, err = await download_bytes(original_path)
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
    
    acknowledged_ids = request.acknowledged_issue_ids
    
    print(f"🔵 [APP] Processing job {job_id} with acknowledged_ids: {acknowledged_ids}")
    
    # Optimization: Skip re-analysis if acknowledged_ids is empty and analysis already exists
    if not acknowledged_ids and current_job.get("final_status"):
        print(f"🔵 [APP] Using cached analysis result for {job_id}")
        analyze_result = current_job
    elif USE_V2_ANALYZER:
        # Re-run V2 analysis with acknowledged_ids
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
    
    try:
        # Load image - reuse already downloaded image_bytes or read from local file
        if image_bytes:
            # Already downloaded from Supabase earlier
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Local file - read and cache bytes
            image_bgr = cv2.imread(saved_file_path)
            if image_bgr is not None:
                # Read bytes for later use
                with open(saved_file_path, "rb") as f:
                    image_bytes = f.read()
        
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
            # Upload processed image to Supabase Storage (use async versions in async endpoint)
            processed_key = get_object_key("processed", job_id, ".png")
            stored_key, storage_error = await upload_bytes(processed_key, processed_bytes, "image/png")
            
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
                processed_url, _ = await create_signed_url(processed_key, 86400)
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
        
        # Store in processing cache for the brief analysis window (no bytes!)
        _processing_jobs[job_id] = {
            "status": "processing",
            "result": None,
            "reasons": [],
            "preview_url": preview_url
        }
        
        # Add job to queue (not direct BackgroundTasks - prevents overload)
        # Queue processes jobs sequentially to prevent Render health check failures
        job_queue = get_job_queue()
        path_to_process = object_key if use_supabase else original_image_path
        queued = job_queue.enqueue(job_id, path_to_process, ext)
        
        if not queued:
            print(f"⚠️ [UPLOAD] Job {job_id} already in queue")
        
        # Get queue stats for response
        queue_stats = job_queue.get_queue_stats()
        queue_position = queue_stats.get("queue_size", 0)
        
        # Başarılı yükleme - JSON response
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "preview_url": preview_url,
            "storage": "supabase" if use_supabase else "local",
            "queue_position": queue_position,
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
    
    Checks both PayTR (DB) and Stripe (in-memory) payment status.
    
    If no payment system is configured:
    - Returns enabled=false
    - can_download=true (free download during beta)
    """
    # Check if any payment system is enabled
    paytr_enabled = is_paytr_configured()
    stripe_enabled = is_payments_enabled()
    any_payments_enabled = paytr_enabled or stripe_enabled
    
    # Get job from DB to check PayTR payment state
    db_job, db_error = await get_job_safe(job_id)
    
    if db_error:
        return JSONResponse({
            "error": "Veritabanı hatası"
        }, status_code=503)
    
    if not db_job:
        return JSONResponse({
            "error": "İş bulunamadı"
        }, status_code=404)
    
    # Check PayTR payment state from DB
    db_payment_state = db_job.get("payment_state", "")
    db_paid = db_payment_state == "PAID"
    
    # Check Stripe in-memory order (legacy)
    stripe_order = get_order(job_id)
    stripe_paid = is_paid(job_id) if stripe_enabled else False
    
    # Combined paid status
    paid = db_paid or stripe_paid
    
    # If no payment system is configured, allow free download
    if not any_payments_enabled:
        return JSONResponse({
            "enabled": False,
            "paid": True,  # Treat as paid when payments disabled
            "state": "FREE",
            "can_download": True,
            "download_url": generate_signed_url(job_id),
            "message": "Beta döneminde indirme ücretsiz"
        })
    
    # Determine state
    if paid:
        state = "PAID"
    elif db_payment_state == "PAYMENT_PENDING":
        state = "PAYMENT_PENDING"
    elif db_payment_state == "FAILED":
        state = "FAILED"
    elif stripe_order:
        state = stripe_order.get("state", "ANALYZED")
    else:
        state = "ANALYZED"
    
    return JSONResponse({
        "enabled": True,
        "paid": paid,
        "state": state,
        "payment_provider": "paytr" if db_paid else ("stripe" if stripe_paid else None),
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
    
    Payment verification:
    - Checks job.payment_state in DB (for PayTR)
    - Falls back to Stripe in-memory order store
    - DEV_ALLOW_FREE_DOWNLOADS env var can bypass payment check
    """
    # Check DEV_ALLOW_FREE_DOWNLOADS flag
    dev_free_downloads = os.getenv("DEV_ALLOW_FREE_DOWNLOADS", "").lower() == "true"
    
    # If signed URL parameters provided, verify them
    if expires is not None and sig is not None:
        if not verify_signed_url(job_id, expires, sig):
            raise HTTPException(status_code=403, detail="Geçersiz veya süresi dolmuş link")
    elif not dev_free_downloads:
        # No signed URL - verify payment status
        # First check DB for PayTR payment state
        db_job, _ = await get_job_safe(job_id)
        db_paid = db_job and db_job.get("payment_state") == "PAID" if db_job else False
        
        # Also check Stripe in-memory store (legacy)
        stripe_paid = is_paid(job_id) if is_payments_enabled() else False
        
        # Check PayTR configured
        paytr_configured = is_paytr_configured()
        stripe_configured = is_payments_enabled()
        
        # If neither payment system is configured, allow download (beta mode)
        if not paytr_configured and not stripe_configured:
            print(f"[DOWNLOAD] job_id={job_id} - No payment system configured, allowing download")
        elif not db_paid and not stripe_paid:
            print(f"[DOWNLOAD] job_id={job_id} paid=false - blocking download")
            raise HTTPException(status_code=402, detail="İndirme için ödeme gerekli")
        else:
            print(f"[DOWNLOAD] job_id={job_id} paid=true")
    
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


@app.get("/api/test/background-task", response_class=JSONResponse)
async def test_background_task(background_tasks: BackgroundTasks):
    """
    Test endpoint to verify background tasks are working.
    """
    import time as t
    
    test_id = f"test-{int(t.time())}"
    
    def simple_bg_task():
        print(f"🔵 [TEST_BG] Background task {test_id} started")
        t.sleep(2)
        print(f"🔵 [TEST_BG] Background task {test_id} completed")
    
    background_tasks.add_task(simple_bg_task)
    
    return JSONResponse({
        "ok": True,
        "test_id": test_id,
        "message": "Background task scheduled - check logs"
    })


@app.post("/api/test/process/{job_id}", response_class=JSONResponse)
async def test_process_sync(job_id: str):
    """
    Debug endpoint: Trigger process_job_with_path synchronously.
    Returns immediately with result or error.
    """
    # Get job from DB to find object_key
    db_job, db_error = await get_job_safe(job_id)
    
    if db_error or not db_job:
        return JSONResponse({
            "ok": False,
            "error": f"Job not found: {db_error or 'No such job'}"
        }, status_code=404)
    
    original_path = db_job.get("original_image_path", "")
    if not original_path:
        return JSONResponse({
            "ok": False,
            "error": "Job has no original_image_path"
        }, status_code=400)
    
    # Determine extension from path
    ext = os.path.splitext(original_path)[1] or ".jpg"
    
    print(f"🧪 [TEST] Running process_job_with_path synchronously for {job_id}")
    
    try:
        result = process_job_with_path(job_id, original_path, ext)
        return JSONResponse({
            "ok": True,
            "db_saved": result,
            "message": "Processing completed synchronously"
        })
    except Exception as e:
        import traceback
        return JSONResponse({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-500:]
        }, status_code=500)


@app.get("/api/test/psycopg2", response_class=JSONResponse)
async def test_psycopg2():
    """Test psycopg2 sync connection."""
    import psycopg2
    import re
    from utils.env_config import get_database_url
    
    try:
        database_url, _ = get_database_url(required=False)
        if not database_url:
            return JSONResponse({"ok": False, "error": "No DATABASE_URL"})
        
        psycopg_url = re.sub(r'^postgres://', 'postgresql://', database_url)
        
        conn = psycopg2.connect(psycopg_url, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SELECT 1, NOW()::text")
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return JSONResponse({"ok": True, "result": list(result)})
    except Exception as e:
        import traceback
        return JSONResponse({
            "ok": False, 
            "error": str(e)[:200],
            "traceback": traceback.format_exc()[-500:]
        })


@app.post("/api/test/sync-analyze", response_class=JSONResponse)
async def test_sync_analyze():
    """
    Test endpoint to verify analyzer and DB work synchronously.
    Creates a blank test image, analyzes it, and saves result to DB.
    """
    import tempfile
    import cv2
    import numpy as np
    import psycopg2
    import json
    
    from utils.analyze_v2 import analyze_image_v2
    from utils.env_config import get_database_url
    
    results = {
        "steps": [],
        "ok": False
    }
    
    # Step 1: Create test image
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:] = (255, 255, 255)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, test_img)
            temp_path = tmp.name
        
        results["steps"].append({"step": "create_image", "ok": True, "path": temp_path})
    except Exception as e:
        results["steps"].append({"step": "create_image", "ok": False, "error": str(e)})
        return JSONResponse(results)
    
    # Step 2: Analyze image
    try:
        analyze_result = analyze_image_v2("sync-test", temp_path)
        results["steps"].append({
            "step": "analyze", 
            "ok": True, 
            "final_status": analyze_result.get("final_status"),
            "issues": [i.get("id") for i in analyze_result.get("issues", [])]
        })
    except Exception as e:
        results["steps"].append({"step": "analyze", "ok": False, "error": str(e)})
        return JSONResponse(results)
    finally:
        import os
        os.unlink(temp_path)
    
    # Step 3: Test DB connection with psycopg2
    try:
        database_url, _ = get_database_url(required=False)
        if not database_url:
            results["steps"].append({"step": "db", "ok": False, "error": "No DATABASE_URL"})
            return JSONResponse(results)
        
        # Convert to psycopg2 format
        import re
        psycopg_url = re.sub(r'^postgres://', 'postgresql://', database_url)
        
        conn = psycopg2.connect(psycopg_url)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        results["steps"].append({"step": "db", "ok": True, "result": result[0]})
    except Exception as e:
        results["steps"].append({"step": "db", "ok": False, "error": str(e)[:200]})
        return JSONResponse(results)
    
    results["ok"] = True
    return JSONResponse(results)


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


# ============================================================================
# PayTR Payment Endpoints
# ============================================================================

class PayTRCustomer(BaseModel):
    email: str
    full_name: Optional[str] = "Müşteri"
    phone: Optional[str] = "5000000000"
    address: Optional[str] = "Türkiye"
    city: Optional[str] = "İstanbul"
    district: Optional[str] = ""
    postal_code: Optional[str] = "34000"


class PayTRInitRequest(BaseModel):
    job_id: str
    product_type: str  # "digital" or "digital_print"
    customer: PayTRCustomer
    ack_ids: Optional[List[str]] = []


@app.get("/api/config/paytr", response_class=JSONResponse)
async def get_paytr_config_endpoint():
    """
    Get PayTR payment configuration for frontend.
    Returns enabled status and prices.
    """
    config = get_paytr_config()
    return JSONResponse(config)


@app.post("/api/paytr/init", response_class=JSONResponse)
async def paytr_init(request: Request, init_req: PayTRInitRequest):
    """
    Initialize PayTR payment session.
    
    Creates iframe token and payment record in database.
    
    Request JSON:
    {
        "job_id": "uuid",
        "product_type": "digital" | "digital_print",
        "customer": {
            "email": "user@example.com",
            "full_name": "Ad Soyad",
            "phone": "5XXXXXXXXX",
            "address": "Adres",
            "city": "Şehir",
            "postal_code": "34000"
        },
        "ack_ids": ["GLASSES_DETECTED"]  // optional, for acknowledged warnings
    }
    
    Returns:
    {
        "success": true,
        "token": "paytr_iframe_token",
        "merchant_oid": "job_xxx_timestamp",
        "amount_kurus": 10000,
        "checkout_url": "/payment/paytr?token=xxx"
    }
    """
    job_id = init_req.job_id
    product_type = init_req.product_type
    customer = init_req.customer
    ack_ids = init_req.ack_ids or []
    
    # Validate PayTR is configured
    if not is_paytr_configured():
        return JSONResponse({
            "success": False,
            "error": "Ödeme sistemi yapılandırılmamış",
            "code": "PAYTR_NOT_CONFIGURED"
        }, status_code=501)
    
    # Validate product type
    if product_type not in ["digital", "digital_print"]:
        return JSONResponse({
            "success": False,
            "error": "Geçersiz ürün tipi"
        }, status_code=400)
    
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
    
    # Check can_continue
    can_continue = db_job.get("can_continue", False)
    requires_ack_ids = db_job.get("requires_ack_ids", [])
    
    # If requires acknowledgement, check if provided
    if requires_ack_ids and not can_continue:
        # Check if all required acks are provided
        missing_acks = [ack for ack in requires_ack_ids if ack not in ack_ids]
        if missing_acks:
            return JSONResponse({
                "success": False,
                "error": f"Uyarılar onaylanmalı: {', '.join(missing_acks)}",
                "code": "ACK_REQUIRED",
                "missing_acks": missing_acks
            }, status_code=400)
    
    # Get client IP
    user_ip = get_client_ip(request)
    
    # Build URLs
    base_url = str(request.base_url).rstrip("/")
    merchant_ok_url = f"{base_url}/payment/success"
    merchant_fail_url = f"{base_url}/payment/cancel"
    
    # Create iframe token
    token, merchant_oid, paytr_error = create_iframe_token(
        job_id=job_id,
        product_type=product_type,
        user_email=customer.email,
        user_ip=user_ip,
        user_name=customer.full_name or "Müşteri",
        user_address=customer.address or "Türkiye",
        user_phone=customer.phone or "5000000000",
        merchant_ok_url=merchant_ok_url,
        merchant_fail_url=merchant_fail_url
    )
    
    if paytr_error:
        print(f"[PAYTR_INIT] Error: {paytr_error}")
        return JSONResponse({
            "success": False,
            "error": f"Ödeme başlatılamadı: {paytr_error}",
            "code": "PAYTR_ERROR"
        }, status_code=500)
    
    # Determine amount
    amount_kurus = PAYTR_DIGITAL_PRICE if product_type == "digital" else PAYTR_PRINT_PRICE
    
    # Update job with payment info
    try:
        import json
        import asyncpg
        import re
        
        database_url = os.getenv("DATABASE_URL", "")
        if database_url:
            asyncpg_url = re.sub(r'^postgres(ql)?://', 'postgresql://', database_url)
            
            conn = await asyncpg.connect(asyncpg_url)
            try:
                # Upsert payment record
                await conn.execute("""
                    INSERT INTO payments (id, job_id, provider, status, amount_kurus, currency, provider_ref, product_type, created_at, updated_at)
                    VALUES (gen_random_uuid(), $1::uuid, 'paytr', 'INIT', $2, 'TRY', $3, $4, NOW(), NOW())
                    ON CONFLICT (job_id, provider) DO UPDATE SET
                        status = 'INIT',
                        amount_kurus = $2,
                        provider_ref = $3,
                        product_type = $4,
                        updated_at = NOW()
                """, job_id, amount_kurus, merchant_oid, product_type)
                
                # Update job payment state and email
                await conn.execute("""
                    UPDATE jobs 
                    SET payment_state = 'PAYMENT_PENDING',
                        user_email = $2,
                        updated_at = NOW()
                    WHERE id = $1::uuid
                """, job_id, customer.email)
                
                # If digital_print, create/update print_orders
                if product_type == "digital_print":
                    await conn.execute("""
                        INSERT INTO print_orders (id, job_id, full_name, phone, email, address, city, district, postal_code, status, created_at, updated_at)
                        VALUES (gen_random_uuid(), $1::uuid, $2, $3, $4, $5, $6, $7, $8, 'CREATED', NOW(), NOW())
                        ON CONFLICT (job_id) DO UPDATE SET
                            full_name = $2,
                            phone = $3,
                            email = $4,
                            address = $5,
                            city = $6,
                            district = $7,
                            postal_code = $8,
                            status = 'CREATED',
                            updated_at = NOW()
                    """, job_id, customer.full_name, customer.phone, customer.email,
                        customer.address, customer.city, customer.district or "", customer.postal_code)
                
                print(f"[PAYTR_INIT] DB updated for job {job_id}")
            finally:
                await conn.close()
    except Exception as e:
        print(f"[PAYTR_INIT] DB error (non-fatal): {e}")
        # Continue anyway - payment can still work
    
    print(f"[PAYTR_INIT] Success - job_id={job_id}, merchant_oid={merchant_oid}, amount={amount_kurus}")
    
    return JSONResponse({
        "success": True,
        "token": token,
        "merchant_oid": merchant_oid,
        "amount_kurus": amount_kurus,
        "checkout_url": f"/payment/paytr?token={token}&merchant_oid={merchant_oid}&product_type={product_type}"
    })


@app.get("/payment/paytr", response_class=HTMLResponse)
async def paytr_checkout_page(request: Request, token: str, merchant_oid: str = "", product_type: str = "digital"):
    """
    Render PayTR checkout page with embedded iframe.
    """
    if not token:
        return templates.TemplateResponse("payment_cancel.html", {
            "request": request,
            "job_id": None,
            "error": "Geçersiz ödeme tokeni"
        })
    
    # Determine product name and amount for display
    if product_type == "digital_print":
        product_name = "Dijital + Baskı (4 adet)"
        amount_kurus = PAYTR_PRINT_PRICE
    else:
        product_name = "Dijital Fotoğraf"
        amount_kurus = PAYTR_DIGITAL_PRICE
    
    amount_display = f"₺{amount_kurus / 100:.0f}"
    
    return templates.TemplateResponse("paytr_checkout.html", {
        "request": request,
        "token": token,
        "merchant_oid": merchant_oid,
        "product_name": product_name,
        "amount_display": amount_display,
        "product_type": product_type
    })


@app.post("/api/paytr/webhook")
async def paytr_webhook(request: Request):
    """
    Handle PayTR webhook notifications.
    
    PayTR sends POST with form data:
    - merchant_oid: Order ID
    - status: "success" or "failed"
    - total_amount: Amount in kuruş
    - hash: Verification hash
    - failed_reason_code: Error code (if failed)
    - failed_reason_msg: Error message (if failed)
    - test_mode: "1" if test transaction
    - payment_type: "card" or "eft"
    - currency: "TL", "USD", etc.
    - payment_amount: Original amount
    
    Must respond with "OK" text.
    """
    try:
        form_data = await request.form()
        
        merchant_oid = form_data.get("merchant_oid", "")
        status = form_data.get("status", "")
        total_amount = form_data.get("total_amount", "")
        received_hash = form_data.get("hash", "")
        failed_reason_code = form_data.get("failed_reason_code", "")
        failed_reason_msg = form_data.get("failed_reason_msg", "")
        test_mode = form_data.get("test_mode", "0")
        payment_type = form_data.get("payment_type", "card")
        
        print(f"[PAYTR_WEBHOOK] merchant_oid={merchant_oid}, status={status}, amount={total_amount}")
        
        # Verify hash
        if not verify_webhook_hash(merchant_oid, status, total_amount, received_hash):
            print(f"[PAYTR_WEBHOOK] Invalid hash for {merchant_oid}")
            return HTMLResponse("INVALID_HASH", status_code=400)
        
        # Parse job_id from merchant_oid
        job_id = parse_merchant_oid(merchant_oid)
        if not job_id:
            print(f"[PAYTR_WEBHOOK] Invalid merchant_oid format: {merchant_oid}")
            # Still return OK to prevent retries
            return HTMLResponse("OK")
        
        # Check idempotency - if already processed, just return OK
        try:
            import asyncpg
            import re
            
            database_url = os.getenv("DATABASE_URL", "")
            if database_url:
                asyncpg_url = re.sub(r'^postgres(ql)?://', 'postgresql://', database_url)
                
                conn = await asyncpg.connect(asyncpg_url)
                try:
                    # Check if already processed
                    existing = await conn.fetchrow("""
                        SELECT webhook_processed_at FROM payments 
                        WHERE provider_ref = $1 AND provider = 'paytr'
                    """, merchant_oid)
                    
                    if existing and existing['webhook_processed_at']:
                        print(f"[PAYTR_WEBHOOK] Already processed: {merchant_oid}")
                        return HTMLResponse("OK")
                    
                    if status == "success":
                        # Update payment status
                        await conn.execute("""
                            UPDATE payments 
                            SET status = 'PAID',
                                webhook_processed_at = NOW(),
                                updated_at = NOW()
                            WHERE provider_ref = $1 AND provider = 'paytr'
                        """, merchant_oid)
                        
                        # Update job payment state
                        await conn.execute("""
                            UPDATE jobs 
                            SET payment_state = 'PAID',
                                updated_at = NOW()
                            WHERE id = $1::uuid
                        """, job_id)
                        
                        # If print order, update status
                        await conn.execute("""
                            UPDATE print_orders 
                            SET status = 'PAID',
                                updated_at = NOW()
                            WHERE job_id = $1::uuid
                        """, job_id)
                        
                        print(f"[PAYTR_WEBHOOK] Payment SUCCESS for job {job_id}")
                    else:
                        # Payment failed
                        await conn.execute("""
                            UPDATE payments 
                            SET status = 'FAILED',
                                webhook_processed_at = NOW(),
                                updated_at = NOW()
                            WHERE provider_ref = $1 AND provider = 'paytr'
                        """, merchant_oid)
                        
                        # Update job payment state
                        await conn.execute("""
                            UPDATE jobs 
                            SET payment_state = 'FAILED',
                                updated_at = NOW()
                            WHERE id = $1::uuid
                        """, job_id)
                        
                        print(f"[PAYTR_WEBHOOK] Payment FAILED for job {job_id}: {failed_reason_code} - {failed_reason_msg}")
                    
                finally:
                    await conn.close()
        except Exception as e:
            print(f"[PAYTR_WEBHOOK] DB error: {e}")
            # Still return OK - PayTR will retry otherwise
        
        # Must return "OK" as plain text
        return HTMLResponse("OK")
        
    except Exception as e:
        print(f"[PAYTR_WEBHOOK] Exception: {e}")
        return HTMLResponse("OK")  # Return OK to prevent infinite retries


@app.get("/payment/success", response_class=HTMLResponse)
async def payment_success_page(request: Request, merchant_oid: str = None, session_id: str = None):
    """
    Handle successful payment redirect.
    
    Supports both PayTR (merchant_oid) and Stripe (session_id) parameters.
    """
    job_id = None
    
    # Try PayTR first
    if merchant_oid:
        job_id = parse_merchant_oid(merchant_oid)
        print(f"[PAYTR_SUCCESS_PAGE] merchant_oid={merchant_oid}, job_id={job_id}")
    
    # Fallback to Stripe
    if not job_id and session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            job_id = session.metadata.get("job_id")
        except:
            pass
    
    return templates.TemplateResponse("payment_success.html", {
        "request": request,
        "session_id": session_id or "",
        "job_id": job_id or "",
        "merchant_oid": merchant_oid or ""
    })


@app.get("/payment/cancel", response_class=HTMLResponse)
async def payment_cancel_page(request: Request, job_id: str = None, merchant_oid: str = None):
    """
    Handle cancelled/failed payment redirect.
    
    Supports both PayTR and Stripe parameters.
    """
    # Try to get job_id from merchant_oid if not provided
    if not job_id and merchant_oid:
        job_id = parse_merchant_oid(merchant_oid)
    
    return templates.TemplateResponse("payment_cancel.html", {
        "request": request,
        "job_id": job_id,
        "merchant_oid": merchant_oid or ""
    })
