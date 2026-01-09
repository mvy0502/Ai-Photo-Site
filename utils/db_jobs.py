"""
Database Repository Layer for Jobs
Provides async CRUD operations for the jobs table.

Production-hardened:
- DB-first reads (no memory precedence)
- Memory fallback only when DEV_ALLOW_MEMORY_FALLBACK=true
- Atomic updates with automatic updated_at
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import text

from .db import db_manager


# ============================================================================
# Configuration
# ============================================================================

# Only allow memory fallback in dev mode (explicit opt-in)
DEV_ALLOW_MEMORY_FALLBACK = os.getenv("DEV_ALLOW_MEMORY_FALLBACK", "false").lower() == "true"

# In-memory storage (only used when DEV_ALLOW_MEMORY_FALLBACK=true)
_memory_jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Constants
# ============================================================================

class JobStatus:
    """Job lifecycle states."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    ANALYZED = "ANALYZED"
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class PaymentState:
    """Payment lifecycle states."""
    ANALYZED = "ANALYZED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAID = "PAID"
    DELIVERED = "DELIVERED"


# ============================================================================
# Exceptions
# ============================================================================

class DatabaseError(Exception):
    """Raised when database operation fails."""
    pass


class JobNotFoundError(Exception):
    """Raised when job is not found in database."""
    pass


# ============================================================================
# Helpers
# ============================================================================

def _serialize_json(value: Any) -> str:
    """Safely serialize value to JSON string."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _row_to_dict(row) -> Optional[Dict[str, Any]]:
    """Convert SQLAlchemy row to dictionary."""
    if row is None:
        return None
    return dict(row._mapping)


def _is_db_available() -> bool:
    """Check if database is connected."""
    return db_manager.connection_info.get("connected", False)


def _get_memory_fallback(job_id: str) -> Optional[Dict[str, Any]]:
    """Get from memory only if DEV_ALLOW_MEMORY_FALLBACK is enabled."""
    if DEV_ALLOW_MEMORY_FALLBACK:
        job = _memory_jobs.get(job_id)
        if job:
            print(f"⚠️ [DEV] Using memory fallback for job {job_id}")
        return job
    return None


# ============================================================================
# Core DB Operations (no fallback - strict)
# ============================================================================

async def create_job(
    job_id: Optional[str] = None,
    original_image_path: Optional[str] = None,
    status: str = JobStatus.PROCESSING,
    payment_state: str = PaymentState.ANALYZED
) -> Dict[str, Any]:
    """
    Create a new job record in the database.
    
    Args:
        job_id: Optional UUID string. If not provided, one will be generated.
        original_image_path: Path to the original uploaded image.
        status: Initial job status (default: PROCESSING).
        payment_state: Initial payment state (default: ANALYZED).
    
    Returns:
        Dictionary with job data including id.
    
    Raises:
        DatabaseError: If database is not connected or insert fails.
    """
    if not _is_db_available():
        raise DatabaseError("Database not connected")
    
    job_uuid = job_id or str(uuid4())
    
    try:
        async with db_manager.engine.begin() as conn:
            result = await conn.execute(
                text("""
                    INSERT INTO jobs (id, status, original_image_path, payment_state,
                                      acknowledged_issue_ids, requires_ack_ids)
                    VALUES (:id, :status, :original_image_path, :payment_state,
                            '[]'::jsonb, '[]'::jsonb)
                    RETURNING id, status, created_at, updated_at, original_image_path, 
                              can_continue, payment_state
                """),
                {
                    "id": job_uuid,
                    "status": status,
                    "original_image_path": original_image_path,
                    "payment_state": payment_state
                }
            )
            row = result.fetchone()
            return _row_to_dict(row)
    except Exception as e:
        raise DatabaseError(f"Failed to create job: {e}")


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a job by ID from database.
    
    Args:
        job_id: UUID string of the job.
    
    Returns:
        Job dictionary or None if not found.
    
    Raises:
        DatabaseError: If database is not connected or query fails.
    """
    if not _is_db_available():
        raise DatabaseError("Database not connected")
    
    try:
        async with db_manager.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT id, status, created_at, updated_at,
                           original_image_path, normalized_image_path, processed_image_path,
                           analysis_result, final_image_mime,
                           requires_ack_ids, acknowledged_issue_ids,
                           can_continue, payment_state, user_email
                    FROM jobs
                    WHERE id = :id
                """),
                {"id": job_id}
            )
            row = result.fetchone()
            return _row_to_dict(row)
    except Exception as e:
        raise DatabaseError(f"Failed to get job: {e}")


async def update_job(job_id: str, **fields) -> Optional[Dict[str, Any]]:
    """
    Atomically update job fields. Updated_at is set automatically.
    
    Args:
        job_id: UUID string of the job.
        **fields: Fields to update (status, analysis_result, processed_image_path, etc.)
    
    Returns:
        Updated job dictionary or None if not found.
    
    Raises:
        DatabaseError: If database is not connected or update fails.
    """
    if not _is_db_available():
        raise DatabaseError("Database not connected")
    
    if not fields:
        return await get_job(job_id)
    
    # Build dynamic SET clause
    set_clauses = []
    params = {"id": job_id}
    
    for key, value in fields.items():
        # JSONB fields need explicit cast
        if key in ("analysis_result", "requires_ack_ids", "acknowledged_issue_ids"):
            set_clauses.append(f"{key} = :{key}::jsonb")
            params[key] = _serialize_json(value)
        else:
            set_clauses.append(f"{key} = :{key}")
            params[key] = value
    
    # updated_at is handled by trigger, but we can also set it explicitly
    query = f"""
        UPDATE jobs 
        SET {', '.join(set_clauses)}
        WHERE id = :id
        RETURNING id, status, created_at, updated_at,
                  original_image_path, normalized_image_path, processed_image_path,
                  analysis_result, requires_ack_ids, acknowledged_issue_ids,
                  can_continue, payment_state, user_email
    """
    
    try:
        async with db_manager.engine.begin() as conn:
            result = await conn.execute(text(query), params)
            row = result.fetchone()
            return _row_to_dict(row)
    except Exception as e:
        raise DatabaseError(f"Failed to update job: {e}")


async def save_analysis_result(
    job_id: str,
    status: str,
    analysis_result: Dict[str, Any],
    normalized_image_path: Optional[str] = None,
    requires_ack_ids: Optional[List[str]] = None,
    can_continue: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Save analysis result to a job. Does not touch processed_image_path.
    
    Args:
        job_id: UUID string of the job.
        status: Final status (PASS, WARN, FAIL).
        analysis_result: Full analysis output as dictionary.
        normalized_image_path: Path to normalized/preview image.
        requires_ack_ids: List of issue IDs requiring acknowledgement.
        can_continue: Whether user can proceed.
    
    Returns:
        Updated job dictionary or None if not found.
    """
    return await update_job(
        job_id,
        status=status,
        analysis_result=analysis_result,
        normalized_image_path=normalized_image_path,
        requires_ack_ids=requires_ack_ids or [],
        acknowledged_issue_ids=[],  # Reset on new analysis
        can_continue=can_continue
    )


async def save_processed_image(
    job_id: str,
    processed_image_path: str,
    normalized_image_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Save processed image path after PhotoRoom processing.
    Does not overwrite analysis_result.
    
    Args:
        job_id: UUID string of the job.
        processed_image_path: Path to the processed image.
        normalized_image_path: Optional path to normalized image.
    
    Returns:
        Updated job dictionary or None if not found.
    """
    fields = {"processed_image_path": processed_image_path}
    if normalized_image_path:
        fields["normalized_image_path"] = normalized_image_path
    return await update_job(job_id, **fields)


async def set_acknowledged_issue_ids(
    job_id: str,
    acknowledged_ids: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Set which issues have been acknowledged by user.
    Recalculates can_continue based on requires_ack_ids.
    
    Args:
        job_id: UUID string of the job.
        acknowledged_ids: List of acknowledged issue IDs.
    
    Returns:
        Updated job dictionary or None if not found.
    """
    # First get the job to check requires_ack_ids
    job = await get_job(job_id)
    if not job:
        return None
    
    requires_ack = job.get("requires_ack_ids") or []
    status = job.get("status", "")
    
    # can_continue = no FAIL status AND all required acks are acknowledged
    has_fail = status == JobStatus.FAIL
    all_acked = all(req_id in acknowledged_ids for req_id in requires_ack)
    can_continue = not has_fail and (not requires_ack or all_acked)
    
    return await update_job(
        job_id,
        acknowledged_issue_ids=acknowledged_ids,
        can_continue=can_continue
    )


async def update_payment_state(
    job_id: str,
    payment_state: str,
    user_email: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Update payment state for a job.
    
    Args:
        job_id: UUID string of the job.
        payment_state: New payment state.
        user_email: Optional user email for delivery.
    
    Returns:
        Updated job dictionary or None if not found.
    """
    fields = {"payment_state": payment_state}
    if user_email:
        fields["user_email"] = user_email
    return await update_job(job_id, **fields)


async def is_job_paid(job_id: str) -> bool:
    """
    Check if a job has been paid for.
    
    Args:
        job_id: UUID string of the job.
    
    Returns:
        True if payment_state is PAID or DELIVERED.
    """
    job = await get_job(job_id)
    if not job:
        return False
    return job.get("payment_state") in [PaymentState.PAID, PaymentState.DELIVERED]


# ============================================================================
# Production-safe wrappers (DB-first, optional memory fallback for dev)
# ============================================================================

async def get_job_safe(job_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Get job from DB with proper error handling.
    
    Returns:
        Tuple of (job_dict, error_message).
        - If job found: (job_dict, None)
        - If job not found: (None, None)
        - If DB error and no fallback: (None, error_message)
        - If DB error with DEV fallback: (memory_job or None, None)
    """
    try:
        job = await get_job(job_id)
        return (job, None)
    except DatabaseError as e:
        error_msg = str(e)
        # Only use memory fallback if explicitly enabled for dev
        if DEV_ALLOW_MEMORY_FALLBACK:
            memory_job = _memory_jobs.get(job_id)
            if memory_job:
                print(f"⚠️ [DEV] DB error, using memory fallback: {error_msg[:50]}")
                return (memory_job, None)
        # Production: return error
        return (None, error_msg)


async def create_job_safe(
    job_id: Optional[str] = None,
    original_image_path: Optional[str] = None,
    status: str = JobStatus.PROCESSING
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Create job in DB with proper error handling.
    
    Returns:
        Tuple of (job_dict, error_message).
    """
    try:
        job = await create_job(job_id, original_image_path, status)
        return (job, None)
    except DatabaseError as e:
        error_msg = str(e)
        # Only use memory fallback if explicitly enabled for dev
        if DEV_ALLOW_MEMORY_FALLBACK:
            job_uuid = job_id or str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            memory_job = {
                "id": job_uuid,
                "status": status,
                "created_at": now,
                "updated_at": now,
                "original_image_path": original_image_path,
                "normalized_image_path": None,
                "processed_image_path": None,
                "analysis_result": None,
                "requires_ack_ids": [],
                "acknowledged_issue_ids": [],
                "can_continue": False,
                "payment_state": PaymentState.ANALYZED,
                "user_email": None,
                "_in_memory": True
            }
            _memory_jobs[job_uuid] = memory_job
            print(f"⚠️ [DEV] DB error, using memory fallback: {error_msg[:50]}")
            return (memory_job, None)
        return (None, error_msg)


async def update_job_safe(job_id: str, **fields) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update job in DB with proper error handling.
    
    Returns:
        Tuple of (job_dict, error_message).
    """
    try:
        job = await update_job(job_id, **fields)
        return (job, None)
    except DatabaseError as e:
        error_msg = str(e)
        if DEV_ALLOW_MEMORY_FALLBACK and job_id in _memory_jobs:
            _memory_jobs[job_id].update(fields)
            _memory_jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            print(f"⚠️ [DEV] DB error, using memory fallback: {error_msg[:50]}")
            return (_memory_jobs[job_id], None)
        return (None, error_msg)


# ============================================================================
# Cleanup Operations
# ============================================================================

async def delete_old_jobs(days: int = 7, dry_run: bool = True) -> Tuple[int, List[str]]:
    """
    Delete jobs older than specified days.
    
    Args:
        days: Number of days after which jobs are considered old.
        dry_run: If True, only return what would be deleted without deleting.
    
    Returns:
        Tuple of (count, list of deleted job IDs).
    """
    if not _is_db_available():
        raise DatabaseError("Database not connected")
    
    try:
        async with db_manager.engine.begin() as conn:
            # First get the jobs that would be deleted
            result = await conn.execute(
                text("""
                    SELECT id, original_image_path, processed_image_path
                    FROM jobs
                    WHERE created_at < NOW() - INTERVAL ':days days'
                """.replace(":days", str(int(days))))
            )
            rows = result.fetchall()
            job_ids = [str(row[0]) for row in rows]
            paths = [(row[1], row[2]) for row in rows]
            
            if dry_run:
                return (len(job_ids), job_ids)
            
            # Delete the jobs
            if job_ids:
                await conn.execute(
                    text("""
                        DELETE FROM jobs
                        WHERE created_at < NOW() - INTERVAL ':days days'
                    """.replace(":days", str(int(days))))
                )
            
            return (len(job_ids), job_ids)
    except Exception as e:
        raise DatabaseError(f"Failed to delete old jobs: {e}")
