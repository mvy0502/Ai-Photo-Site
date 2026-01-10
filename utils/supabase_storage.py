"""
Supabase Storage Client Wrapper

Provides functions to upload, download, and create signed URLs for files
stored in Supabase Storage buckets.

Environment Variables:
    SUPABASE_URL: Project URL (https://xxxx.supabase.co)
    SUPABASE_SERVICE_ROLE_KEY: Service role key (server-side only)
    SUPABASE_STORAGE_BUCKET: Bucket name (default: "photos")
    SUPABASE_STORAGE_SIGNED_URL_TTL: TTL in seconds (default: 86400 = 24h)
"""

import os
from typing import Optional, Tuple
from functools import lru_cache

# Lazy imports to avoid startup failures when env vars not set
_supabase_client = None


def _get_config() -> dict:
    """Get Supabase configuration from environment."""
    return {
        "url": os.getenv("SUPABASE_URL", ""),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        "bucket": os.getenv("SUPABASE_STORAGE_BUCKET", "photos"),
        "signed_url_ttl": int(os.getenv("SUPABASE_STORAGE_SIGNED_URL_TTL", "86400")),
    }


def is_storage_configured() -> bool:
    """Check if Supabase Storage is properly configured."""
    config = _get_config()
    return bool(config["url"] and config["service_role_key"])


def get_storage_client():
    """
    Get or create the Supabase client.
    Returns None if not configured.
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    config = _get_config()
    
    if not config["url"] or not config["service_role_key"]:
        print("⚠️ [STORAGE] Supabase Storage not configured (missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)")
        return None
    
    try:
        from supabase import create_client, Client
        _supabase_client = create_client(
            config["url"],
            config["service_role_key"]
        )
        print(f"✅ [STORAGE] Supabase client initialized for bucket: {config['bucket']}")
        return _supabase_client
    except Exception as e:
        print(f"❌ [STORAGE] Failed to initialize Supabase client: {e}")
        return None


def get_object_key(prefix: str, job_id: str, extension: str = ".jpg") -> str:
    """
    Generate a deterministic object key.
    
    Args:
        prefix: Folder prefix (e.g., "originals", "processed", "normalized")
        job_id: Unique job identifier
        extension: File extension (e.g., ".jpg", ".png")
    
    Returns:
        Object key like "originals/abc123.jpg"
    """
    # Sanitize extension
    if not extension.startswith("."):
        extension = f".{extension}"
    
    # Sanitize job_id (should be UUID, but be safe)
    safe_job_id = "".join(c for c in job_id if c.isalnum() or c == "-")
    
    return f"{prefix}/{safe_job_id}{extension}"


async def upload_bytes(
    object_key: str,
    content: bytes,
    content_type: str = "image/jpeg"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Upload bytes to Supabase Storage.
    
    Args:
        object_key: Path in bucket (e.g., "originals/job123.jpg")
        content: File content as bytes
        content_type: MIME type (e.g., "image/jpeg", "image/png")
    
    Returns:
        Tuple of (object_key, error_message)
        On success: (object_key, None)
        On failure: (None, error_message)
    """
    client = get_storage_client()
    if not client:
        return None, "Storage not configured"
    
    config = _get_config()
    bucket = config["bucket"]
    
    try:
        # Use upsert=True to overwrite if exists
        result = client.storage.from_(bucket).upload(
            path=object_key,
            file=content,
            file_options={
                "content-type": content_type,
                "upsert": "true"  # Overwrite if exists
            }
        )
        
        print(f"✅ [STORAGE] Uploaded: {bucket}/{object_key} ({len(content)} bytes)")
        return object_key, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [STORAGE] Upload failed for {object_key}: {error_msg}")
        return None, error_msg


async def create_signed_url(
    object_key: str,
    ttl_seconds: Optional[int] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Create a signed URL for downloading a file.
    
    Args:
        object_key: Path in bucket (e.g., "processed/job123.png")
        ttl_seconds: Time-to-live in seconds (default from env)
    
    Returns:
        Tuple of (signed_url, error_message)
        On success: (url, None)
        On failure: (None, error_message)
    """
    client = get_storage_client()
    if not client:
        return None, "Storage not configured"
    
    config = _get_config()
    bucket = config["bucket"]
    ttl = ttl_seconds or config["signed_url_ttl"]
    
    try:
        result = client.storage.from_(bucket).create_signed_url(
            path=object_key,
            expires_in=ttl
        )
        
        signed_url = result.get("signedURL") or result.get("signedUrl")
        
        if signed_url:
            print(f"✅ [STORAGE] Signed URL created for {object_key} (TTL: {ttl}s)")
            return signed_url, None
        else:
            return None, "No signed URL in response"
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [STORAGE] Signed URL failed for {object_key}: {error_msg}")
        return None, error_msg


async def download_bytes(object_key: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download file content as bytes.
    
    Args:
        object_key: Path in bucket
    
    Returns:
        Tuple of (content_bytes, error_message)
    """
    client = get_storage_client()
    if not client:
        return None, "Storage not configured"
    
    config = _get_config()
    bucket = config["bucket"]
    
    try:
        result = client.storage.from_(bucket).download(object_key)
        
        if result:
            print(f"✅ [STORAGE] Downloaded: {bucket}/{object_key} ({len(result)} bytes)")
            return result, None
        else:
            return None, "Empty response"
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [STORAGE] Download failed for {object_key}: {error_msg}")
        return None, error_msg


async def delete_object(object_key: str) -> Tuple[bool, Optional[str]]:
    """
    Delete an object from storage.
    
    Args:
        object_key: Path in bucket
    
    Returns:
        Tuple of (success, error_message)
    """
    client = get_storage_client()
    if not client:
        return False, "Storage not configured"
    
    config = _get_config()
    bucket = config["bucket"]
    
    try:
        client.storage.from_(bucket).remove([object_key])
        print(f"✅ [STORAGE] Deleted: {bucket}/{object_key}")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [STORAGE] Delete failed for {object_key}: {error_msg}")
        return False, error_msg


def get_content_type(extension: str) -> str:
    """Get MIME type for file extension."""
    ext = extension.lower().lstrip(".")
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "application/octet-stream")


# Magic bytes for image validation
IMAGE_MAGIC_BYTES = {
    b'\xff\xd8\xff': 'image/jpeg',
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'RIFF': 'image/webp',  # WebP starts with RIFF....WEBP
}


def validate_image_bytes(content: bytes, expected_type: str = None) -> Tuple[bool, str, str]:
    """
    Validate image content by checking magic bytes.
    
    Args:
        content: File content as bytes
        expected_type: Expected MIME type (optional)
    
    Returns:
        Tuple of (is_valid, detected_type, error_message)
    """
    if len(content) < 8:
        return False, "", "File too small to be a valid image"
    
    detected_type = None
    
    # Check magic bytes
    for magic, mime_type in IMAGE_MAGIC_BYTES.items():
        if content[:len(magic)] == magic:
            detected_type = mime_type
            break
    
    # Special case for WebP (check for WEBP after RIFF)
    if detected_type == "image/webp" and b"WEBP" not in content[:12]:
        detected_type = None
    
    if not detected_type:
        return False, "", "Invalid image format (magic bytes check failed)"
    
    if expected_type and detected_type != expected_type:
        return False, detected_type, f"Type mismatch: expected {expected_type}, got {detected_type}"
    
    return True, detected_type, ""


# Sync wrappers for use in non-async contexts
def upload_bytes_sync(object_key: str, content: bytes, content_type: str = "image/jpeg") -> Tuple[Optional[str], Optional[str]]:
    """Synchronous wrapper for upload_bytes."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(upload_bytes(object_key, content, content_type))


def create_signed_url_sync(object_key: str, ttl_seconds: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
    """Synchronous wrapper for create_signed_url."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(create_signed_url(object_key, ttl_seconds))


def download_bytes_sync(object_key: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Synchronous wrapper for download_bytes."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(download_bytes(object_key))
