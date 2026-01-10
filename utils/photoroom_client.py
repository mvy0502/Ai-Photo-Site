"""
PhotoRoom Background Removal API Client

Production-ready client for PhotoRoom Remove Background API.
Supports both sandbox and live modes via environment variables.

Environment Variables:
    PHOTOROOM_API_KEY: API key (required)
    PHOTOROOM_BASE_URL: Base URL (optional, defaults to production)
    DEBUG_BG: Set to "1" to enable debug output
"""

import os
import requests
from typing import Optional, Tuple
import time

# =============================================================================
# Configuration
# =============================================================================

# API endpoints
PHOTOROOM_PRODUCTION_URL = "https://sdk.photoroom.com/v1/segment"
PHOTOROOM_SANDBOX_URL = "https://sandbox-api.photoroom.com/v1/segment"  # If different

# Default settings
DEFAULT_SIZE = "full"  # Keep original resolution
DEFAULT_BG_COLOR = "f5f6f8"  # Biometric light gray (without #)

# Debug flag
DEBUG_ENABLED = os.getenv("DEBUG_BG", "0") == "1"


def _get_api_key() -> Optional[str]:
    """Get PhotoRoom API key from environment (strip whitespace/newlines)"""
    key = os.getenv("PHOTOROOM_API_KEY")
    if key:
        key = key.strip()
    return key if key else None


def _get_base_url() -> str:
    """Get PhotoRoom API base URL from environment or use production default"""
    return os.getenv("PHOTOROOM_BASE_URL", PHOTOROOM_PRODUCTION_URL)


# =============================================================================
# Main API Functions
# =============================================================================

def remove_background_photoroom(
    image_bytes: bytes,
    *,
    bg_color: Optional[str] = None,
    size: str = DEFAULT_SIZE,
    crop: bool = False,
    timeout: int = 60
) -> Tuple[bytes, dict]:
    """
    Remove background from image using PhotoRoom API.
    
    IMPORTANT: Never use crop=True - we handle cropping ourselves to preserve
    hair and shoulders in biometric photos.
    
    Args:
        image_bytes: Raw image bytes (JPEG/PNG)
        bg_color: Background color hex (without #), e.g., "f5f6f8"
                  If None, returns transparent PNG (RGBA)
        size: Output size - "full" (original), "hd" (1920px), "medium" (1000px)
        crop: MUST BE FALSE - we never let PhotoRoom crop
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (image_bytes, metadata_dict)
        
    Raises:
        ValueError: If API key not configured or invalid parameters
        RuntimeError: If API request fails
    """
    # Validate parameters
    if crop:
        raise ValueError("crop=True is not allowed. We handle cropping ourselves to preserve hair/shoulders.")
    
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "PhotoRoom API key not configured. "
            "Set PHOTOROOM_API_KEY environment variable."
        )
    
    if len(image_bytes) == 0:
        raise ValueError("Empty image bytes provided")
    
    # Prepare request
    api_url = _get_base_url()
    
    headers = {
        "x-api-key": api_key
    }
    
    files = {
        "image_file": ("image.jpg", image_bytes, "image/jpeg")
    }
    
    # Build form data
    # Note: PhotoRoom API parameters
    data = {
        "size": size,
        "format": "png",  # Always PNG for alpha channel
        "crop": "false",  # NEVER crop - we handle it
    }
    
    # Add background color if specified
    # If not specified, PhotoRoom returns transparent PNG
    if bg_color:
        # PhotoRoom expects hex without #
        clean_color = bg_color.lstrip("#")
        data["bg_color"] = clean_color
    
    if DEBUG_ENABLED:
        print(f"[PhotoRoom] Sending request to {api_url}")
        print(f"[PhotoRoom] Image size: {len(image_bytes)} bytes")
        print(f"[PhotoRoom] Parameters: size={size}, bg_color={bg_color}, crop={crop}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=timeout
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"PhotoRoom API request timed out after {timeout}s")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Could not connect to PhotoRoom API: {e}")
    
    elapsed = time.time() - start_time
    
    # Check response
    if response.status_code != 200:
        error_msg = f"PhotoRoom API error: HTTP {response.status_code}"
        try:
            error_detail = response.json()
            error_msg += f" - {error_detail}"
        except:
            error_msg += f" - {response.text[:500]}"
        
        if DEBUG_ENABLED:
            print(f"[PhotoRoom] ERROR: {error_msg}")
        
        raise RuntimeError(error_msg)
    
    # Success
    result_bytes = response.content
    
    metadata = {
        "success": True,
        "input_size_bytes": len(image_bytes),
        "output_size_bytes": len(result_bytes),
        "processing_time_ms": round(elapsed * 1000, 1),
        "bg_color": bg_color,
        "size": size,
        "api_url": api_url,
        "has_transparency": bg_color is None
    }
    
    if DEBUG_ENABLED:
        print(f"[PhotoRoom] Success! Output: {len(result_bytes)} bytes in {elapsed*1000:.0f}ms")
    
    return result_bytes, metadata


def check_api_configuration() -> dict:
    """
    Check PhotoRoom API configuration status.
    
    Returns:
        Dict with configuration status (without exposing full key)
    """
    api_key = _get_api_key()
    base_url = _get_base_url()
    
    return {
        "api_configured": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_prefix": api_key[:10] + "..." if api_key and len(api_key) > 10 else None,
        "base_url": base_url,
        "is_sandbox": "sandbox" in base_url.lower() if base_url else False,
        "default_bg_color": f"#{DEFAULT_BG_COLOR}",
        "default_size": DEFAULT_SIZE
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def remove_background_transparent(
    image_bytes: bytes,
    *,
    size: str = DEFAULT_SIZE,
    timeout: int = 60
) -> Tuple[bytes, dict]:
    """
    Remove background and return transparent PNG.
    
    Use this when you want to composite onto custom background yourself.
    """
    return remove_background_photoroom(
        image_bytes,
        bg_color=None,  # No bg_color = transparent
        size=size,
        crop=False,
        timeout=timeout
    )


def remove_background_biometric(
    image_bytes: bytes,
    *,
    size: str = DEFAULT_SIZE,
    timeout: int = 60
) -> Tuple[bytes, dict]:
    """
    Remove background and replace with biometric light gray (#f5f6f8).
    
    This is the standard background for passport/ID photos.
    """
    return remove_background_photoroom(
        image_bytes,
        bg_color=DEFAULT_BG_COLOR,
        size=size,
        crop=False,
        timeout=timeout
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("PhotoRoom API Client - Configuration Check")
    print("=" * 50)
    
    config = check_api_configuration()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if not config["api_configured"]:
        print("\n⚠️  API key not configured!")
        print("   Set PHOTOROOM_API_KEY environment variable")
        sys.exit(1)
    
    print("\n✅ API configured and ready")

