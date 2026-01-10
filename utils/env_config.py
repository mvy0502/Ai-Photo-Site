"""
Environment Configuration Helper

Provides robust parsing and validation of environment variables,
handling common issues like trailing newlines, whitespace, and format validation.

Usage:
    from utils.env_config import get_env, get_database_url, get_supabase_url, get_config_summary
"""

import os
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urlunparse


class ConfigError(Exception):
    """Raised when a required configuration is missing or invalid."""
    pass


def get_env(
    name: str,
    default: Optional[str] = None,
    required: bool = False,
    strip: bool = True
) -> Optional[str]:
    """
    Get environment variable with robust sanitization.
    
    Args:
        name: Environment variable name
        default: Default value if not set
        required: Raise ConfigError if missing/empty
        strip: Strip whitespace/newlines (default True)
    
    Returns:
        Sanitized value or default
    
    Raises:
        ConfigError: If required and missing/empty after sanitization
    """
    value = os.getenv(name, "")
    
    if strip and value:
        # Remove all leading/trailing whitespace including newlines
        value = value.strip()
        # Also remove any embedded newlines (shouldn't happen but be safe)
        value = value.replace('\n', '').replace('\r', '')
    
    if not value:
        if required:
            raise ConfigError(f"Required environment variable '{name}' is not set or empty")
        return default
    
    return value


def sanitize_url(url: str) -> str:
    """
    Sanitize a URL by removing whitespace and trailing slashes.
    """
    if not url:
        return url
    
    # Strip whitespace and newlines
    url = url.strip().replace('\n', '').replace('\r', '')
    
    # Remove trailing slash
    url = url.rstrip('/')
    
    return url


def validate_database_url(url: str) -> tuple[str, list[str]]:
    """
    Validate and sanitize DATABASE_URL.
    
    Returns:
        Tuple of (sanitized_url, list_of_warnings)
    """
    warnings = []
    
    if not url:
        return url, ["DATABASE_URL is empty"]
    
    original = url
    
    # Sanitize
    url = url.strip().replace('\n', '').replace('\r', '').replace('\t', '')
    
    # Check for URL-encoded newline (%0A)
    if '%0A' in url or '%0a' in url:
        url = url.replace('%0A', '').replace('%0a', '')
        warnings.append("DATABASE_URL contained URL-encoded newline (%0A), removed")
    
    # Check if it changed
    if url != original:
        warnings.append("DATABASE_URL contained whitespace/newlines, sanitized")
    
    # Parse URL
    try:
        parsed = urlparse(url)
        
        # Check for pooler recommendations
        if 'pooler.supabase.com' in (parsed.hostname or ''):
            if parsed.port == 5432:
                warnings.append(
                    "DATABASE_URL uses port 5432 with Supabase pooler. "
                    "Consider using port 6543 (session pooler) for better compatibility with Render."
                )
        
        # Check scheme
        if parsed.scheme not in ('postgres', 'postgresql'):
            warnings.append(f"DATABASE_URL has unusual scheme: {parsed.scheme}")
        
    except Exception as e:
        warnings.append(f"DATABASE_URL parsing error: {e}")
    
    return url, warnings


def get_database_url(required: bool = True) -> tuple[Optional[str], list[str]]:
    """
    Get and validate DATABASE_URL.
    
    Returns:
        Tuple of (sanitized_url, list_of_warnings)
    """
    raw_url = os.getenv("DATABASE_URL", "")
    
    if not raw_url:
        if required:
            return None, ["DATABASE_URL is not set"]
        return None, []
    
    return validate_database_url(raw_url)


def validate_supabase_url(url: str) -> tuple[str, list[str]]:
    """
    Validate and sanitize SUPABASE_URL.
    
    Expected format: https://<project-ref>.supabase.co
    
    Returns:
        Tuple of (sanitized_url, list_of_warnings)
    """
    warnings = []
    
    if not url:
        return url, []
    
    original = url
    
    # Sanitize
    url = sanitize_url(url)
    
    if url != original:
        warnings.append("SUPABASE_URL contained whitespace/newlines/trailing slash, sanitized")
    
    # Validate format
    pattern = r'^https://[a-z0-9-]+\.supabase\.co$'
    if not re.match(pattern, url):
        # Try to fix common issues
        if url.startswith('http://'):
            url = url.replace('http://', 'https://')
            warnings.append("SUPABASE_URL used http://, changed to https://")
        
        if not url.startswith('https://'):
            warnings.append("SUPABASE_URL should start with https://")
        
        if not url.endswith('.supabase.co'):
            warnings.append("SUPABASE_URL should end with .supabase.co")
    
    return url, warnings


def get_supabase_url(required: bool = False) -> tuple[Optional[str], list[str]]:
    """
    Get and validate SUPABASE_URL.
    
    Returns:
        Tuple of (sanitized_url, list_of_warnings)
    """
    raw_url = os.getenv("SUPABASE_URL", "")
    
    if not raw_url:
        if required:
            return None, ["SUPABASE_URL is not set"]
        return None, []
    
    return validate_supabase_url(raw_url)


def extract_project_ref(supabase_url: str) -> Optional[str]:
    """
    Extract project reference from SUPABASE_URL.
    
    Example: https://abc123xyz.supabase.co -> abc123xyz
    """
    if not supabase_url:
        return None
    
    try:
        parsed = urlparse(supabase_url)
        hostname = parsed.hostname or ""
        if hostname.endswith('.supabase.co'):
            return hostname.replace('.supabase.co', '')
    except:
        pass
    
    return None


def mask_url_credentials(url: str) -> str:
    """
    Mask username and password in a URL for safe logging.
    
    Example: postgresql://user:pass@host:5432/db -> postgresql://***:***@host:5432/db
    """
    if not url:
        return url
    
    try:
        parsed = urlparse(url)
        
        # Rebuild with masked credentials
        if parsed.username or parsed.password:
            # Replace credentials
            netloc = f"***:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            
            masked = urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            return masked
        
        return url
    except:
        # If parsing fails, do a simple regex replacement
        return re.sub(r'://[^@]+@', '://***:***@', url)


def get_config_summary() -> Dict[str, Any]:
    """
    Get a non-secret configuration summary for debugging.
    
    Returns dict with:
    - db_configured: bool
    - db_host: masked host info
    - db_port: port number
    - db_warnings: list of warnings
    - storage_configured: bool
    - storage_bucket: bucket name
    - supabase_project_ref: project reference
    - supabase_warnings: list of warnings
    - payments_enabled: bool
    - photoroom_configured: bool
    """
    summary = {
        "db_configured": False,
        "db_host": None,
        "db_port": None,
        "db_warnings": [],
        "storage_configured": False,
        "storage_bucket": None,
        "supabase_project_ref": None,
        "supabase_warnings": [],
        "payments_enabled": False,
        "photoroom_configured": False,
    }
    
    # Database
    db_url, db_warnings = get_database_url(required=False)
    summary["db_warnings"] = db_warnings
    
    if db_url:
        summary["db_configured"] = True
        try:
            parsed = urlparse(db_url)
            summary["db_host"] = parsed.hostname
            summary["db_port"] = parsed.port
        except:
            pass
    
    # Supabase Storage
    supabase_url, supabase_warnings = get_supabase_url(required=False)
    summary["supabase_warnings"] = supabase_warnings
    
    service_key = get_env("SUPABASE_SERVICE_ROLE_KEY")
    bucket = get_env("SUPABASE_STORAGE_BUCKET", default="photos")
    
    # Track which vars are missing for clearer debugging
    storage_missing = []
    if not supabase_url:
        storage_missing.append("SUPABASE_URL")
    if not service_key:
        storage_missing.append("SUPABASE_SERVICE_ROLE_KEY")
    
    summary["storage_missing_vars"] = storage_missing
    summary["storage_bucket"] = bucket  # Always show bucket config
    
    if supabase_url and service_key:
        summary["storage_configured"] = True
        summary["supabase_project_ref"] = extract_project_ref(supabase_url)
    else:
        summary["supabase_project_ref"] = extract_project_ref(supabase_url) if supabase_url else None
    
    # Payments
    stripe_secret = get_env("STRIPE_SECRET_KEY")
    stripe_pub = get_env("STRIPE_PUBLISHABLE_KEY")
    stripe_webhook = get_env("STRIPE_WEBHOOK_SECRET")
    payments_enabled_flag = get_env("PAYMENTS_ENABLED", default="true").lower()
    
    if stripe_secret and stripe_pub and stripe_webhook and payments_enabled_flag != "false":
        summary["payments_enabled"] = True
    
    # PhotoRoom
    photoroom_key = get_env("PHOTOROOM_API_KEY")
    summary["photoroom_configured"] = bool(photoroom_key)
    
    return summary


def validate_all_config() -> tuple[bool, list[str]]:
    """
    Validate all configuration on startup.
    
    Returns:
        Tuple of (all_valid, list_of_messages)
    """
    messages = []
    all_valid = True
    
    # Database
    db_url, db_warnings = get_database_url(required=False)
    if db_warnings:
        messages.extend([f"⚠️ DB: {w}" for w in db_warnings])
    if db_url:
        messages.append(f"✅ DATABASE_URL configured (host: {urlparse(db_url).hostname})")
    else:
        messages.append("⚠️ DATABASE_URL not set - app may have limited functionality")
    
    # Supabase
    supabase_url, supabase_warnings = get_supabase_url(required=False)
    if supabase_warnings:
        messages.extend([f"⚠️ Supabase: {w}" for w in supabase_warnings])
    
    service_key = get_env("SUPABASE_SERVICE_ROLE_KEY")
    if supabase_url and service_key:
        project_ref = extract_project_ref(supabase_url)
        messages.append(f"✅ Supabase Storage configured (project: {project_ref})")
    elif supabase_url or service_key:
        messages.append("⚠️ Supabase Storage partially configured - need both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    else:
        messages.append("ℹ️ Supabase Storage not configured - using local filesystem (not persistent on Render)")
    
    # PhotoRoom
    photoroom_key = get_env("PHOTOROOM_API_KEY")
    if photoroom_key:
        messages.append("✅ PhotoRoom API configured")
    else:
        messages.append("⚠️ PHOTOROOM_API_KEY not set - image processing will be limited")
    
    # Payments
    summary = get_config_summary()
    if summary["payments_enabled"]:
        messages.append("✅ Payments enabled (Stripe)")
    else:
        messages.append("ℹ️ Payments disabled - free downloads enabled")
    
    return all_valid, messages


# Run validation on import (optional - can be disabled)
def startup_validation():
    """Run startup validation and print results."""
    print("=" * 60)
    print("🔧 Configuration Validation")
    print("=" * 60)
    
    _, messages = validate_all_config()
    for msg in messages:
        print(f"  {msg}")
    
    print("=" * 60)
