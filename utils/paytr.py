"""
PayTR Payment Integration Module

Implements PayTR iFrame API for payment processing.
Based on PayTR documentation: https://dev.paytr.com/iframe-api

Flow:
1. Backend calls PayTR API to get iframe_token
2. Frontend embeds PayTR iframe with token
3. PayTR sends webhook to /api/paytr/webhook on payment completion
4. User is redirected to success/fail URL

Environment Variables:
- PAYTR_MERCHANT_ID: Merchant ID from PayTR panel
- PAYTR_MERCHANT_KEY: Merchant Key from PayTR panel  
- PAYTR_MERCHANT_SALT: Merchant Salt from PayTR panel
- PAYTR_TEST_MODE: "true" for test mode (default: true)
"""

import os
import base64
import hmac
import hashlib
import json
import time
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
PAYTR_MERCHANT_ID = os.getenv("PAYTR_MERCHANT_ID", "")
PAYTR_MERCHANT_KEY = os.getenv("PAYTR_MERCHANT_KEY", "").encode('utf-8')
PAYTR_MERCHANT_SALT = os.getenv("PAYTR_MERCHANT_SALT", "").encode('utf-8')
PAYTR_TEST_MODE = os.getenv("PAYTR_TEST_MODE", "true").lower() == "true"

# PayTR API endpoint
PAYTR_API_URL = "https://www.paytr.com/odeme/api/get-token"
PAYTR_IFRAME_URL = "https://www.paytr.com/odeme/guvenli"

# Pricing in kuruş (100 TL = 10000 kuruş)
DIGITAL_PRICE_KURUS = int(os.getenv("DIGITAL_PRICE_KURUS", "10000"))  # 100 TL
PRINT_PRICE_KURUS = int(os.getenv("PRINT_PRICE_KURUS", "40000"))  # 400 TL

# Currency
CURRENCY = "TL"


# ============================================================================
# PayTR Feature Flag
# ============================================================================
def is_paytr_configured() -> bool:
    """
    Check if PayTR is properly configured.
    
    Returns True only if all required env vars are present:
    - PAYTR_MERCHANT_ID
    - PAYTR_MERCHANT_KEY
    - PAYTR_MERCHANT_SALT
    """
    required_vars = [
        PAYTR_MERCHANT_ID,
        PAYTR_MERCHANT_KEY,
        PAYTR_MERCHANT_SALT
    ]
    return all(var and len(var) > 0 for var in required_vars)


def get_paytr_config() -> Dict[str, Any]:
    """
    Get PayTR configuration for frontend.
    Safe to call even when PayTR is not configured.
    """
    configured = is_paytr_configured()
    
    return {
        "enabled": configured,
        "test_mode": PAYTR_TEST_MODE,
        "prices": {
            "digital": {
                "amount_kurus": DIGITAL_PRICE_KURUS,
                "amount_tl": DIGITAL_PRICE_KURUS / 100,
                "display": f"₺{DIGITAL_PRICE_KURUS / 100:.0f}",
                "name": "Dijital",
                "description": "Anında indir",
            },
            "digital_print": {
                "amount_kurus": PRINT_PRICE_KURUS,
                "amount_tl": PRINT_PRICE_KURUS / 100,
                "display": f"₺{PRINT_PRICE_KURUS / 100:.0f}",
                "name": "Dijital + Baskı",
                "description": "4 adet baskı + ücretsiz kargo",
            }
        }
    }


# ============================================================================
# Token Generation
# ============================================================================
def generate_merchant_oid(job_id: str) -> str:
    """
    Generate unique merchant order ID.
    Format: job_<job_id>_<timestamp>
    
    Must be unique for each transaction and max 64 chars.
    """
    timestamp = int(time.time())
    # job_id is UUID (36 chars), timestamp is ~10 chars, total ~50 chars
    return f"job_{job_id}_{timestamp}"


def build_paytr_token(
    merchant_oid: str,
    user_ip: str,
    email: str,
    payment_amount: int,
    user_basket: str,
    no_installment: str = "1",
    max_installment: str = "0",
    currency: str = "TL",
    test_mode: str = "1"
) -> str:
    """
    Build PayTR token for authentication.
    
    Token formula (from PayTR docs):
    hash_str = merchant_id + user_ip + merchant_oid + email + payment_amount + 
               user_basket + no_installment + max_installment + currency + test_mode
    paytr_token = base64(hmac_sha256(hash_str + merchant_salt, merchant_key))
    
    Args:
        merchant_oid: Unique order ID
        user_ip: Customer IP address
        email: Customer email
        payment_amount: Amount in kuruş (e.g., 10000 for 100 TL)
        user_basket: Base64 encoded basket JSON
        no_installment: "1" to disable installments
        max_installment: "0" for default max installments
        currency: "TL", "USD", "EUR", etc.
        test_mode: "1" for test mode
    
    Returns:
        Base64 encoded HMAC-SHA256 token
    """
    hash_str = (
        PAYTR_MERCHANT_ID +
        user_ip +
        merchant_oid +
        email +
        str(payment_amount) +
        user_basket +
        no_installment +
        max_installment +
        currency +
        test_mode
    )
    
    # HMAC-SHA256 with merchant_key, then base64 encode
    token = base64.b64encode(
        hmac.new(
            PAYTR_MERCHANT_KEY,
            (hash_str.encode('utf-8') + PAYTR_MERCHANT_SALT),
            hashlib.sha256
        ).digest()
    )
    
    return token.decode('utf-8')


def create_user_basket(product_name: str, amount_kurus: int, quantity: int = 1) -> str:
    """
    Create base64 encoded basket JSON for PayTR.
    
    Format: [[product_name, price_str, quantity], ...]
    Price is in TL (not kuruş), as string.
    
    Args:
        product_name: Product name (max 255 chars)
        amount_kurus: Price in kuruş
        quantity: Quantity
    
    Returns:
        Base64 encoded JSON string
    """
    price_tl = f"{amount_kurus / 100:.2f}"
    basket = [[product_name, price_tl, quantity]]
    basket_json = json.dumps(basket)
    return base64.b64encode(basket_json.encode('utf-8')).decode('utf-8')


def get_client_ip(request) -> str:
    """
    Get client IP from request, handling proxies.
    
    Checks X-Forwarded-For header first (for reverse proxy/load balancer),
    then falls back to request.client.host.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        Client IP address string
    """
    # Check X-Forwarded-For header (set by proxies)
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2
        # The first one is the original client
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP header (used by some proxies)
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client IP
    if request.client and request.client.host:
        return request.client.host
    
    return "127.0.0.1"


# ============================================================================
# PayTR API Integration
# ============================================================================
def create_iframe_token(
    job_id: str,
    product_type: str,
    user_email: str,
    user_ip: str,
    user_name: str = "Müşteri",
    user_address: str = "Türkiye",
    user_phone: str = "5000000000",
    merchant_ok_url: str = "",
    merchant_fail_url: str = "",
    timeout_limit: int = 30
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Create PayTR iframe token by calling PayTR API.
    
    Args:
        job_id: Job UUID
        product_type: "digital" or "digital_print"
        user_email: Customer email
        user_ip: Customer IP
        user_name: Customer name (default: "Müşteri")
        user_address: Customer address (default: "Türkiye")
        user_phone: Customer phone (default: "5000000000")
        merchant_ok_url: Success redirect URL
        merchant_fail_url: Failure redirect URL
        timeout_limit: Payment timeout in minutes (default: 30)
    
    Returns:
        Tuple of (token, merchant_oid, error)
        - On success: (token, merchant_oid, None)
        - On failure: (None, None, error_message)
    """
    if not is_paytr_configured():
        return None, None, "PayTR not configured"
    
    # Determine amount based on product type
    if product_type == "digital":
        amount_kurus = DIGITAL_PRICE_KURUS
        product_name = "Türkiye Biyometrik Fotoğraf - Dijital"
    else:  # digital_print
        amount_kurus = PRINT_PRICE_KURUS
        product_name = "Türkiye Biyometrik Fotoğraf - Dijital + Baskı"
    
    # Generate unique merchant order ID
    merchant_oid = generate_merchant_oid(job_id)
    
    # Create basket
    user_basket = create_user_basket(product_name, amount_kurus)
    
    # Test mode
    test_mode = "1" if PAYTR_TEST_MODE else "0"
    
    # Build token
    paytr_token = build_paytr_token(
        merchant_oid=merchant_oid,
        user_ip=user_ip,
        email=user_email,
        payment_amount=amount_kurus,
        user_basket=user_basket,
        no_installment="1",  # No installments for this product
        max_installment="0",
        currency=CURRENCY,
        test_mode=test_mode
    )
    
    # Prepare POST data
    post_data = {
        "merchant_id": PAYTR_MERCHANT_ID,
        "user_ip": user_ip,
        "merchant_oid": merchant_oid,
        "email": user_email,
        "payment_amount": str(amount_kurus),
        "paytr_token": paytr_token,
        "user_basket": user_basket,
        "debug_on": "1",  # Enable debug for development
        "no_installment": "1",
        "max_installment": "0",
        "user_name": user_name[:60],  # Max 60 chars
        "user_address": user_address[:400],  # Max 400 chars
        "user_phone": user_phone[:20],  # Max 20 chars
        "merchant_ok_url": merchant_ok_url[:400],
        "merchant_fail_url": merchant_fail_url[:400],
        "timeout_limit": str(timeout_limit),
        "currency": CURRENCY,
        "test_mode": test_mode,
        "lang": "tr"
    }
    
    print(f"[PAYTR_INIT] job_id={job_id}, product_type={product_type}, amount={amount_kurus}")
    
    try:
        response = requests.post(PAYTR_API_URL, data=post_data, timeout=30)
        result = response.json()
        
        if result.get("status") == "success":
            token = result.get("token")
            print(f"[PAYTR_INIT] Success - merchant_oid={merchant_oid}")
            return token, merchant_oid, None
        else:
            error_reason = result.get("reason", "Unknown error")
            print(f"[PAYTR_INIT] Failed - reason={error_reason}")
            return None, None, error_reason
            
    except requests.exceptions.RequestException as e:
        error_msg = f"PayTR API request failed: {str(e)}"
        print(f"[PAYTR_INIT] Exception - {error_msg}")
        return None, None, error_msg
    except json.JSONDecodeError as e:
        error_msg = f"PayTR API response parse error: {str(e)}"
        print(f"[PAYTR_INIT] Exception - {error_msg}")
        return None, None, error_msg


# ============================================================================
# Webhook Verification
# ============================================================================
def verify_webhook_hash(
    merchant_oid: str,
    status: str,
    total_amount: str,
    received_hash: str
) -> bool:
    """
    Verify PayTR webhook hash to ensure request authenticity.
    
    Hash formula (from PayTR docs):
    hash = base64(hmac_sha256(merchant_oid + merchant_salt + status + total_amount, merchant_key))
    
    Args:
        merchant_oid: Order ID from webhook
        status: Payment status ("success" or "failed")
        total_amount: Total amount in kuruş
        received_hash: Hash received in webhook
    
    Returns:
        True if hash is valid, False otherwise
    """
    if not is_paytr_configured():
        return False
    
    # Build hash string
    hash_str = merchant_oid + PAYTR_MERCHANT_SALT.decode('utf-8') + status + total_amount
    
    # Calculate expected hash
    expected_hash = base64.b64encode(
        hmac.new(
            PAYTR_MERCHANT_KEY,
            hash_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')
    
    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected_hash, received_hash)


def parse_merchant_oid(merchant_oid: str) -> Optional[str]:
    """
    Parse job_id from merchant_oid.
    
    Format: job_<job_id>_<timestamp>
    
    Args:
        merchant_oid: Order ID string
    
    Returns:
        job_id if valid format, None otherwise
    """
    if not merchant_oid or not merchant_oid.startswith("job_"):
        return None
    
    parts = merchant_oid.split("_")
    if len(parts) < 3:
        return None
    
    # job_id is between "job_" and the last "_timestamp"
    # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    job_id = "_".join(parts[1:-1])
    
    # Validate UUID format (basic check)
    if len(job_id) == 36 and job_id.count("-") == 4:
        return job_id
    
    return None


# ============================================================================
# Utility Functions
# ============================================================================
def get_price_display(product_type: str) -> Dict[str, Any]:
    """Get display information for a product type."""
    config = get_paytr_config()
    return config["prices"].get(product_type, config["prices"]["digital"])


def get_iframe_url(token: str) -> str:
    """Get full PayTR iframe URL with token."""
    return f"{PAYTR_IFRAME_URL}/{token}"


# Log configuration status on module load
if is_paytr_configured():
    print(f"✅ [PAYTR] Configured - test_mode={PAYTR_TEST_MODE}")
else:
    print("ℹ️ [PAYTR] Not configured (missing env vars)")
