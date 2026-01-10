"""
Job Queue System for Background Processing

Implements a single-worker queue to prevent overload:
- Only 1 job processes at a time (configurable via MAX_CONCURRENT_JOBS)
- Jobs are queued with states: QUEUED, PROCESSING, DONE, FAILED
- Worker runs in background thread, doesn't block event loop
- Prevents Render health check timeouts and restarts
"""

import os
import threading
import queue
import time
import traceback
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

# Configuration
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "120"))


class JobState(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class QueuedJob:
    job_id: str
    object_key: str
    ext: str
    state: JobState = JobState.QUEUED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobQueue:
    """
    Thread-safe job queue with single worker consumer.
    
    Usage:
        queue = JobQueue()
        queue.start_worker(processor_func)
        queue.enqueue("job-123", "originals/job-123.jpg", ".jpg")
        status = queue.get_status("job-123")
    """
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_JOBS):
        self._queue: queue.Queue = queue.Queue()
        self._jobs: Dict[str, QueuedJob] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._processor: Optional[Callable] = None
        self._running = False
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._semaphore = threading.Semaphore(max_concurrent)
        
        print(f"📦 [QUEUE] Initialized with max_concurrent={max_concurrent}")
    
    def start_worker(self, processor: Callable[[str, str, str], bool]):
        """
        Start the background worker thread.
        
        Args:
            processor: Function that processes a job (job_id, object_key, ext) -> success
        """
        if self._worker_thread and self._worker_thread.is_alive():
            print("⚠️ [QUEUE] Worker already running")
            return
        
        self._processor = processor
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="JobQueueWorker",
            daemon=True
        )
        self._worker_thread.start()
        print(f"🚀 [QUEUE] Worker started (max_concurrent={self._max_concurrent})")
    
    def stop_worker(self):
        """Stop the worker thread gracefully."""
        self._running = False
        # Put a None to unblock the queue
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        print("🛑 [QUEUE] Worker stopped")
    
    def enqueue(self, job_id: str, object_key: str, ext: str) -> bool:
        """
        Add a job to the queue.
        
        Returns True if queued, False if job_id already exists.
        """
        with self._lock:
            if job_id in self._jobs:
                print(f"⚠️ [QUEUE] Job {job_id} already exists, skipping")
                return False
            
            job = QueuedJob(
                job_id=job_id,
                object_key=object_key,
                ext=ext,
                state=JobState.QUEUED
            )
            self._jobs[job_id] = job
        
        self._queue.put(job_id)
        queue_size = self._queue.qsize()
        print(f"📥 [QUEUE] Job {job_id} queued (queue size: {queue_size})")
        return True
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status. Returns None if job not found."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            return {
                "job_id": job.job_id,
                "state": job.state.value,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error": job.error,
                "queue_position": self._get_queue_position(job_id) if job.state == JobState.QUEUED else None
            }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            states = {}
            for job in self._jobs.values():
                states[job.state.value] = states.get(job.state.value, 0) + 1
            
            return {
                "queue_size": self._queue.qsize(),
                "active_jobs": self._active_count,
                "max_concurrent": self._max_concurrent,
                "total_jobs": len(self._jobs),
                "by_state": states
            }
    
    def _get_queue_position(self, job_id: str) -> int:
        """Get position in queue (1-indexed). Returns 0 if not in queue."""
        # This is approximate since we can't peek into the queue
        position = 1
        for jid, job in self._jobs.items():
            if job.state == JobState.QUEUED:
                if jid == job_id:
                    return position
                position += 1
        return 0
    
    def _worker_loop(self):
        """Main worker loop - processes jobs one at a time."""
        print("🔄 [QUEUE] Worker loop started")
        
        while self._running:
            try:
                # Block waiting for a job (with timeout to check _running)
                try:
                    job_id = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check for shutdown signal
                if job_id is None:
                    break
                
                # Acquire semaphore (limits concurrency)
                self._semaphore.acquire()
                self._active_count += 1
                
                try:
                    self._process_job(job_id)
                finally:
                    self._active_count -= 1
                    self._semaphore.release()
                    self._queue.task_done()
                    
            except Exception as e:
                print(f"🔴 [QUEUE] Worker loop error: {e}")
                traceback.print_exc()
        
        print("🔄 [QUEUE] Worker loop ended")
    
    def _process_job(self, job_id: str):
        """Process a single job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                print(f"⚠️ [QUEUE] Job {job_id} not found in registry")
                return
            
            job.state = JobState.PROCESSING
            job.started_at = datetime.utcnow()
        
        print(f"⚙️ [QUEUE] Processing job {job_id}...")
        
        try:
            if not self._processor:
                raise Exception("No processor configured")
            
            # Call the actual processor
            success = self._processor(job_id, job.object_key, job.ext)
            
            with self._lock:
                if success:
                    job.state = JobState.DONE
                    print(f"✅ [QUEUE] Job {job_id} completed successfully")
                else:
                    job.state = JobState.FAILED
                    job.error = "Processor returned failure"
                    print(f"❌ [QUEUE] Job {job_id} failed (processor returned false)")
                job.completed_at = datetime.utcnow()
                
        except Exception as e:
            error_msg = str(e)
            print(f"🔴 [QUEUE] Job {job_id} failed with exception: {error_msg}")
            traceback.print_exc()
            
            with self._lock:
                job.state = JobState.FAILED
                job.error = error_msg[:500]  # Limit error length
                job.completed_at = datetime.utcnow()
    
    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Remove completed/failed jobs older than max_age_seconds."""
        cutoff = datetime.utcnow().timestamp() - max_age_seconds
        
        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.state in (JobState.DONE, JobState.FAILED):
                    if job.completed_at and job.completed_at.timestamp() < cutoff:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
            
            if to_remove:
                print(f"🧹 [QUEUE] Cleaned up {len(to_remove)} old jobs")


# Global singleton
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue singleton."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


def init_job_queue(processor: Callable[[str, str, str], bool]) -> JobQueue:
    """Initialize and start the job queue with a processor."""
    global _job_queue
    _job_queue = JobQueue()
    _job_queue.start_worker(processor)
    return _job_queue
