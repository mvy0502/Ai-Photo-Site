"""
Email service for sending download links.

Features:
- Signed, expiring download URLs
- Rate limiting to prevent spam
- Idempotency for same job+email within window
"""

import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@biyometrikfoto.tr")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "BiyometrikFoto.tr")

# Rate limiting: minimum seconds between emails for same job+email
EMAIL_RATE_LIMIT_SECONDS = int(os.getenv("EMAIL_RATE_LIMIT_SECONDS", "300"))  # 5 minutes

# In-memory rate limit store (replace with Redis in production)
# Structure: { "job_id:email": timestamp }
email_rate_limit_store: Dict[str, float] = {}


def is_email_configured() -> bool:
    """Check if email is properly configured."""
    return bool(SMTP_USER and SMTP_PASS)


def can_send_email(job_id: str, email: str) -> bool:
    """Check if we can send email (rate limiting)."""
    key = f"{job_id}:{email.lower()}"
    last_sent = email_rate_limit_store.get(key, 0)
    
    return time.time() - last_sent >= EMAIL_RATE_LIMIT_SECONDS


def record_email_sent(job_id: str, email: str):
    """Record that an email was sent."""
    key = f"{job_id}:{email.lower()}"
    email_rate_limit_store[key] = time.time()


def send_download_link(
    job_id: str,
    email: str,
    download_url: str,
    base_url: str = "https://biyometrikfoto.tr",
) -> Dict[str, Any]:
    """
    Send download link email to user.
    
    Args:
        job_id: The job identifier
        email: User's email address
        download_url: The signed download URL (relative path)
        base_url: The base URL of the site
    
    Returns:
        {
            "success": bool,
            "message": str,
            "error": str (optional)
        }
    """
    # Check configuration
    if not is_email_configured():
        return {
            "success": False,
            "message": "E-posta servisi yapılandırılmamış",
            "error": "SMTP not configured"
        }
    
    # Rate limiting
    if not can_send_email(job_id, email):
        return {
            "success": False,
            "message": "Bu e-posta adresine yakın zamanda link gönderildi. Lütfen birkaç dakika bekleyin.",
            "error": "RATE_LIMITED"
        }
    
    # Build full URL
    full_download_url = f"{base_url}{download_url}"
    
    # Create email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Biyometrik Fotoğrafınız Hazır - BiyometrikFoto.tr"
    msg["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_FROM}>"
    msg["To"] = email
    
    # Plain text version
    text_content = f"""
Merhaba,

Biyometrik fotoğrafınız hazır ve indirmeye hazır!

İndirme linki: {full_download_url}

Bu link 24 saat geçerlidir.

Teşekkür ederiz,
BiyometrikFoto.tr Ekibi
    """.strip()
    
    # HTML version
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 30px; text-align: center; border-radius: 12px 12px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 12px 12px; }}
        .button {{ display: inline-block; background: #10b981; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 20px 0; }}
        .button:hover {{ background: #059669; }}
        .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
        .warning {{ background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; border-radius: 8px; margin-top: 20px; color: #92400e; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0; font-size: 24px;">✅ Fotoğrafınız Hazır!</h1>
        </div>
        <div class="content">
            <p>Merhaba,</p>
            <p>Biyometrik fotoğrafınız başarıyla işlendi ve indirmeye hazır.</p>
            
            <div style="text-align: center;">
                <a href="{full_download_url}" class="button">📥 Fotoğrafı İndir</a>
            </div>
            
            <div class="warning">
                ⏰ <strong>Önemli:</strong> Bu indirme linki 24 saat geçerlidir.
            </div>
            
            <p style="margin-top: 20px;">
                Fotoğrafınız Türkiye biyometrik standartlarına (50×60 mm) uygun olarak hazırlanmıştır.
            </p>
        </div>
        <div class="footer">
            <p>Bu e-posta BiyometrikFoto.tr tarafından gönderilmiştir.</p>
            <p>© {datetime.now().year} BiyometrikFoto.tr - Tüm hakları saklıdır.</p>
        </div>
    </div>
</body>
</html>
    """.strip()
    
    part1 = MIMEText(text_content, "plain", "utf-8")
    part2 = MIMEText(html_content, "html", "utf-8")
    
    msg.attach(part1)
    msg.attach(part2)
    
    try:
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, email, msg.as_string())
        
        # Record successful send
        record_email_sent(job_id, email)
        
        return {
            "success": True,
            "message": "İndirme linki e-posta adresinize gönderildi."
        }
        
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": "E-posta gönderilemedi. Lütfen daha sonra tekrar deneyin.",
            "error": "SMTP_AUTH_ERROR"
        }
    except smtplib.SMTPException as e:
        return {
            "success": False,
            "message": "E-posta gönderilemedi. Lütfen daha sonra tekrar deneyin.",
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
            "error": str(e)
        }
