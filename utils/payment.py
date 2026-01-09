"""
Payment module for Stripe integration and order management.

State Machine for each job:
- ANALYZED: Checks done, preview available
- PAYMENT_PENDING: User chose package, checkout created
- PAID: Verified via webhook
- DELIVERED: Email sent (optional)

Payment Feature Flag:
- Payments are automatically disabled if STRIPE_* env vars are missing
- Can be explicitly disabled with PAYMENTS_ENABLED=false
- When disabled, app works without payment features
"""

import os
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import json

# ============================================================================
# Configuration
# ============================================================================
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

# Explicit override to disable payments even if keys exist
PAYMENTS_ENABLED_OVERRIDE = os.getenv("PAYMENTS_ENABLED", "").lower()

# Prices in kuruş (100 TL = 10000 kuruş)
DIGITAL_PRICE_KURUS = int(os.getenv("DIGITAL_PRICE_KURUS", "10000"))  # 100 TL
PRINT_PRICE_KURUS = int(os.getenv("PRINT_PRICE_KURUS", "20000"))  # 200 TL

# Signed URL secret for download links
DOWNLOAD_URL_SECRET = os.getenv("DOWNLOAD_URL_SECRET", "default-secret-change-me")


# ============================================================================
# Payment Feature Flag
# ============================================================================
def is_payments_enabled() -> bool:
    """
    Check if payment features are enabled.
    
    Returns True only if:
    1. PAYMENTS_ENABLED is not explicitly set to 'false'
    2. All required Stripe env vars are present:
       - STRIPE_SECRET_KEY
       - STRIPE_PUBLISHABLE_KEY
       - STRIPE_WEBHOOK_SECRET
    
    Returns False otherwise (graceful degradation).
    """
    # Explicit disable
    if PAYMENTS_ENABLED_OVERRIDE == "false":
        return False
    
    # Check all required vars exist and are non-empty
    required_vars = [
        STRIPE_SECRET_KEY,
        STRIPE_PUBLISHABLE_KEY,
        STRIPE_WEBHOOK_SECRET
    ]
    
    return all(var and len(var) > 0 for var in required_vars)


def get_payments_config() -> Dict[str, Any]:
    """
    Get payment configuration for frontend.
    Safe to call even when payments are disabled.
    """
    enabled = is_payments_enabled()
    
    if enabled:
        return {
            "enabled": True,
            "publishable_key": STRIPE_PUBLISHABLE_KEY,
            "digital_price": get_price_display("digital"),
            "print_price": get_price_display("digital_print"),
        }
    else:
        return {
            "enabled": False,
            "message": "Ödeme sistemi yakında aktif olacak",
            "digital_price": get_price_display("digital"),
            "print_price": get_price_display("digital_print"),
        }


# ============================================================================
# Stripe Import (Conditional)
# ============================================================================
stripe = None

if is_payments_enabled():
    try:
        import stripe as _stripe
        stripe = _stripe
        stripe.api_key = STRIPE_SECRET_KEY
        print("✅ [PAYMENTS] Stripe configured and enabled")
    except ImportError:
        print("⚠️ [PAYMENTS] Stripe package not installed, payments disabled")
else:
    print("ℹ️ [PAYMENTS] Payments disabled (missing env vars or PAYMENTS_ENABLED=false)")


# ============================================================================
# Order State Storage (In-memory for now, replace with DB in production)
# ============================================================================
# Structure: { job_id: { state, package_type, stripe_session_id, ... } }
order_store: Dict[str, Dict[str, Any]] = {}

OrderState = Literal["ANALYZED", "PAYMENT_PENDING", "PAID", "DELIVERED"]
PackageType = Literal["digital", "digital_print"]


def get_order(job_id: str) -> Optional[Dict[str, Any]]:
    """Get order by job_id."""
    return order_store.get(job_id)


def create_order(job_id: str, package_type: PackageType = None) -> Dict[str, Any]:
    """Create or update order for a job."""
    if job_id not in order_store:
        order_store[job_id] = {
            "job_id": job_id,
            "state": "ANALYZED",
            "package_type": package_type,
            "stripe_session_id": None,
            "stripe_payment_intent_id": None,
            "amount_kurus": None,
            "currency": "try",
            "created_at": datetime.utcnow().isoformat(),
            "paid_at": None,
            "email_sent_to": None,
            "email_sent_at": None,
        }
    elif package_type:
        order_store[job_id]["package_type"] = package_type
    return order_store[job_id]


def update_order_state(job_id: str, state: OrderState, **kwargs) -> Optional[Dict[str, Any]]:
    """Update order state and additional fields."""
    if job_id not in order_store:
        return None
    
    order_store[job_id]["state"] = state
    for key, value in kwargs.items():
        order_store[job_id][key] = value
    
    return order_store[job_id]


def is_paid(job_id: str) -> bool:
    """Check if job has been paid."""
    # If payments disabled, treat as paid (free download)
    if not is_payments_enabled():
        return True
    
    order = get_order(job_id)
    if not order:
        return False
    return order.get("state") in ["PAID", "DELIVERED"]


# ============================================================================
# Stripe Checkout
# ============================================================================
class PaymentsDisabledError(Exception):
    """Raised when payment operation is attempted but payments are disabled."""
    pass


def create_checkout_session(
    job_id: str,
    package_type: PackageType,
    success_url: str,
    cancel_url: str,
) -> Dict[str, Any]:
    """
    Create a Stripe Checkout Session for one-time payment.
    
    Raises:
        PaymentsDisabledError: If payments are not enabled.
        ValueError: If Stripe API key not configured.
    
    Returns:
        {
            "checkout_url": "https://checkout.stripe.com/...",
            "session_id": "cs_...",
        }
    """
    if not is_payments_enabled():
        raise PaymentsDisabledError("Payments are disabled")
    
    if not stripe:
        raise ValueError("Stripe not available")
    
    # Determine price
    if package_type == "digital":
        amount = DIGITAL_PRICE_KURUS
        product_name = "Biyometrik Fotoğraf - Dijital"
        description = "Türkiye standartlarına uygun biyometrik fotoğraf (dijital indirme)"
    else:  # digital_print
        amount = PRINT_PRICE_KURUS
        product_name = "Biyometrik Fotoğraf - Dijital + Baskı"
        description = "Türkiye standartlarına uygun biyometrik fotoğraf (dijital + baskı & kargo)"
    
    # Create checkout session
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "try",
                "product_data": {
                    "name": product_name,
                    "description": description,
                },
                "unit_amount": amount,
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "job_id": job_id,
            "package_type": package_type,
            "locale": "tr",
        },
        locale="tr",
    )
    
    # Update order state
    create_order(job_id, package_type)
    update_order_state(
        job_id,
        "PAYMENT_PENDING",
        stripe_session_id=session.id,
        amount_kurus=amount,
    )
    
    return {
        "checkout_url": session.url,
        "session_id": session.id,
    }


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify Stripe webhook signature."""
    if not is_payments_enabled():
        return False
    
    if not stripe:
        return False
    
    if not STRIPE_WEBHOOK_SECRET:
        print("⚠️ [STRIPE] Webhook secret not configured")
        return False
    
    try:
        stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
        return True
    except Exception:
        return False


def handle_checkout_completed(session: Dict[str, Any]) -> bool:
    """
    Handle checkout.session.completed webhook event.
    
    Returns True if successfully processed, False otherwise.
    """
    if not is_payments_enabled():
        return False
    
    job_id = session.get("metadata", {}).get("job_id")
    if not job_id:
        print("⚠️ [STRIPE] No job_id in session metadata")
        return False
    
    order = get_order(job_id)
    if not order:
        print(f"⚠️ [STRIPE] Order not found for job_id: {job_id}")
        return False
    
    # Idempotency check: if already paid, do nothing
    if order.get("state") in ["PAID", "DELIVERED"]:
        print(f"✅ [STRIPE] Order {job_id} already paid, skipping")
        return True
    
    # Verify session ID matches
    if order.get("stripe_session_id") != session.get("id"):
        print(f"⚠️ [STRIPE] Session ID mismatch for job_id: {job_id}")
        return False
    
    # Mark as paid
    update_order_state(
        job_id,
        "PAID",
        stripe_payment_intent_id=session.get("payment_intent"),
        paid_at=datetime.utcnow().isoformat(),
    )
    
    print(f"✅ [STRIPE] Order {job_id} marked as PAID")
    return True


# ============================================================================
# Secure Download Links
# ============================================================================
def generate_signed_url(job_id: str, expires_in_seconds: int = 900) -> str:
    """
    Generate a signed URL for secure download.
    
    Default expiry: 15 minutes (900 seconds)
    """
    expires_at = int(time.time()) + expires_in_seconds
    
    # Create signature
    message = f"{job_id}:{expires_at}"
    signature = hmac.new(
        DOWNLOAD_URL_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()[:32]  # Use first 32 chars
    
    return f"/api/download/{job_id}?expires={expires_at}&sig={signature}"


def verify_signed_url(job_id: str, expires: int, signature: str) -> bool:
    """Verify a signed download URL."""
    # Check expiry
    if int(time.time()) > expires:
        return False
    
    # Verify signature
    message = f"{job_id}:{expires}"
    expected_sig = hmac.new(
        DOWNLOAD_URL_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()[:32]
    
    return hmac.compare_digest(signature, expected_sig)


def generate_email_link(job_id: str, expires_in_hours: int = 24) -> str:
    """
    Generate a signed URL for email delivery.
    
    Default expiry: 24 hours
    """
    return generate_signed_url(job_id, expires_in_seconds=expires_in_hours * 3600)


# ============================================================================
# Utility Functions
# ============================================================================
def get_price_display(package_type: PackageType) -> Dict[str, Any]:
    """Get display information for a package."""
    if package_type == "digital":
        return {
            "amount_kurus": DIGITAL_PRICE_KURUS,
            "amount_tl": DIGITAL_PRICE_KURUS / 100,
            "display": f"₺{DIGITAL_PRICE_KURUS / 100:.0f}",
            "name": "Dijital",
            "description": "Anında indir",
        }
    else:
        return {
            "amount_kurus": PRINT_PRICE_KURUS,
            "amount_tl": PRINT_PRICE_KURUS / 100,
            "display": f"₺{PRINT_PRICE_KURUS / 100:.0f}",
            "name": "Dijital + Baskı",
            "description": "Baskı + kargo (Türkiye içi)",
        }


def get_stripe_publishable_key() -> str:
    """Get Stripe publishable key for frontend."""
    if not is_payments_enabled():
        return ""
    return STRIPE_PUBLISHABLE_KEY
