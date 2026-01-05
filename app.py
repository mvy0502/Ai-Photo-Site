from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from uuid import uuid4
from datetime import datetime
from pathlib import Path

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

# Global job status dictionary
job_status: dict[str, dict] = {}

# Templates klasörünü ayarla
templates = Jinja2Templates(directory="templates")

# Static dosyaları mount et
app.mount("/static", StaticFiles(directory="static"), name="static")

# Uploads klasörünü serve et (preview için)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Uploads klasörünü oluştur
if not os.path.exists("uploads"):
    os.makedirs("uploads")

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
    """Background task - job'u işle"""
    # Mevcut job status'u al (preview_url'yi korumak için)
    current_status = job_status.get(job_id, {})
    preview_url = current_status.get("preview_url", None)
    
    # Job status'u processing olarak set et (preview_url'yi koru)
    job_status[job_id] = {
        "status": "processing",
        "result": None,
        "reasons": [],
        "preview_url": preview_url
    }
    
    # Opsiyonel gecikme (UI için)
    time.sleep(1)
    
    # Analiz yap
    analyze_result = analyze_image(job_id, saved_file_path)
    
    # Preview URL'yi koru
    if preview_url:
        analyze_result["preview_url"] = preview_url
        if "overlay" in analyze_result:
            analyze_result["overlay"]["preview_url"] = preview_url
    
    # Job status'u analiz sonucu ile güncelle
    job_status[job_id] = analyze_result

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
    """Job sayfasını render et"""
    # Job var mı kontrol et
    if job_id not in job_status:
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
    """Job status'u JSON olarak döndür"""
    if job_id not in job_status:
        return JSONResponse({
            "status": "not_found"
        })
    
    # Job status'u direkt döndür
    return JSONResponse(job_status[job_id])

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
        
        # Dosya boyutu kontrolü
        if len(content) > MAX_FILE_SIZE:
            return JSONResponse({
                "success": False,
                "error": "Dosya çok büyük, maksimum 8MB",
                "job_id": None
            }, status_code=400)
        
        # job_id üret
        job_id = str(uuid4())
        
        # Dosya uzantısını al
        ext = get_file_extension(photo.filename)
        
        # Dosya adını oluştur: {job_id}{ext}
        saved_file_path = f"uploads/{job_id}{ext}"
        
        # Dosyayı kaydet
        with open(saved_file_path, "wb") as buffer:
            buffer.write(content)
        
        # Preview URL oluştur
        preview_url = f"/uploads/{job_id}{ext}"
        
        # Job status'u başlangıç durumunda oluştur (process_job background task'ta güncelleyecek)
        job_status[job_id] = {
            "status": "processing",
            "result": None,
            "reasons": [],
            "preview_url": preview_url
        }
        
        # Background task ile process_job başlat
        background_tasks.add_task(process_job, job_id, saved_file_path)
        
        # Başarılı yükleme - JSON response
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "preview_url": preview_url,
            "message": "Fotoğraf başarıyla yüklendi"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Dosya kaydedilirken bir sorun oluştu: {str(e)}",
            "job_id": None
        }, status_code=500)
