"""
DB Write Retry System

Job completion should NOT depend on DB writes.
This module provides async retry with exponential backoff for DB operations.

Flow:
1. Job completes (DONE/FAILED) -> stored in memory immediately
2. DB write attempted
3. If DB fails -> schedule retry (up to 5 attempts over ~60s)
4. Status endpoint returns DONE even if db_saved=false
"""

import threading
import time
import queue
import traceback
import json
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 2
MAX_DELAY_SECONDS = 30


@dataclass
class DBWriteTask:
    """A pending DB write operation."""
    job_id: str
    data: Dict[str, Any]
    attempt: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_error: Optional[str] = None
    
    def next_delay(self) -> float:
        """Calculate exponential backoff delay."""
        delay = BASE_DELAY_SECONDS * (2 ** self.attempt)
        return min(delay, MAX_DELAY_SECONDS)


class DBRetryQueue:
    """
    Background queue for retrying failed DB writes.
    
    Does not block job completion - jobs are marked DONE in memory first,
    then DB writes happen asynchronously with retries.
    """
    
    def __init__(self, writer_func: Callable[[str, Dict], bool]):
        """
        Args:
            writer_func: Function that writes to DB. Takes (job_id, data) -> success bool.
        """
        self._queue: queue.Queue = queue.Queue()
        self._writer = writer_func
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Track write status per job
        self._write_status: Dict[str, Dict] = {}
        self._status_lock = threading.Lock()
    
    def start(self):
        """Start the retry worker thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="DBRetryWorker",
            daemon=True
        )
        self._thread.start()
        print("✅ [DB_RETRY] Worker started")
    
    def stop(self):
        """Stop the retry worker gracefully."""
        self._running = False
        self._queue.put(None)  # Signal to stop
        if self._thread:
            self._thread.join(timeout=5)
        print("🛑 [DB_RETRY] Worker stopped")
    
    def schedule_write(self, job_id: str, data: Dict[str, Any]) -> bool:
        """
        Schedule a DB write (first attempt happens immediately).
        
        Returns True if write succeeded immediately, False if queued for retry.
        """
        # Record initial status
        with self._status_lock:
            self._write_status[job_id] = {
                "db_saved": False,
                "db_error": None,
                "attempts": 0,
                "pending": True
            }
        
        # Try immediate write
        try:
            success = self._writer(job_id, data)
            if success:
                with self._status_lock:
                    self._write_status[job_id] = {
                        "db_saved": True,
                        "db_error": None,
                        "attempts": 1,
                        "pending": False
                    }
                return True
        except Exception as e:
            print(f"⚠️ [DB_RETRY] Immediate write failed for {job_id}: {e}")
        
        # Queue for retry
        task = DBWriteTask(job_id=job_id, data=data, attempt=1, last_error=str(e) if 'e' in dir() else "Unknown")
        self._queue.put(task)
        
        with self._status_lock:
            self._write_status[job_id]["attempts"] = 1
            self._write_status[job_id]["db_error"] = task.last_error
        
        print(f"📥 [DB_RETRY] Queued {job_id} for retry (attempt 1/{MAX_RETRIES})")
        return False
    
    def get_write_status(self, job_id: str) -> Dict[str, Any]:
        """Get the DB write status for a job."""
        with self._status_lock:
            return self._write_status.get(job_id, {
                "db_saved": False,
                "db_error": "No status recorded",
                "attempts": 0,
                "pending": False
            })
    
    def _worker_loop(self):
        """Process retry queue with exponential backoff."""
        print("🔄 [DB_RETRY] Worker loop started")
        
        while self._running:
            try:
                # Get next task (with timeout to check _running)
                try:
                    task = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if task is None:
                    break
                
                # Wait for backoff delay
                delay = task.next_delay()
                print(f"⏳ [DB_RETRY] Waiting {delay}s before retry for {task.job_id}")
                time.sleep(delay)
                
                # Attempt write
                try:
                    success = self._writer(task.job_id, task.data)
                    
                    if success:
                        with self._status_lock:
                            self._write_status[task.job_id] = {
                                "db_saved": True,
                                "db_error": None,
                                "attempts": task.attempt + 1,
                                "pending": False
                            }
                        print(f"✅ [DB_RETRY] Write succeeded for {task.job_id} (attempt {task.attempt + 1})")
                        continue
                    else:
                        task.last_error = "Writer returned False"
                        
                except Exception as e:
                    task.last_error = str(e)[:200]
                    print(f"❌ [DB_RETRY] Attempt {task.attempt + 1} failed for {task.job_id}: {task.last_error}")
                
                # Update status
                with self._status_lock:
                    self._write_status[task.job_id]["attempts"] = task.attempt + 1
                    self._write_status[task.job_id]["db_error"] = task.last_error
                
                # Check if we should retry
                task.attempt += 1
                if task.attempt < MAX_RETRIES:
                    self._queue.put(task)
                    print(f"🔄 [DB_RETRY] Re-queued {task.job_id} (attempt {task.attempt + 1}/{MAX_RETRIES})")
                else:
                    # Max retries reached
                    with self._status_lock:
                        self._write_status[task.job_id]["pending"] = False
                    print(f"💀 [DB_RETRY] Giving up on {task.job_id} after {MAX_RETRIES} attempts")
                    
            except Exception as e:
                print(f"🔴 [DB_RETRY] Worker error: {e}")
                traceback.print_exc()
        
        print("🔄 [DB_RETRY] Worker loop ended")


# Global instance
_retry_queue: Optional[DBRetryQueue] = None


def get_db_retry_queue() -> Optional[DBRetryQueue]:
    """Get the global retry queue (must be initialized first)."""
    return _retry_queue


def init_db_retry_queue(writer_func: Callable[[str, Dict], bool]) -> DBRetryQueue:
    """Initialize the global retry queue."""
    global _retry_queue
    _retry_queue = DBRetryQueue(writer_func)
    _retry_queue.start()
    return _retry_queue


def stop_db_retry_queue():
    """Stop the global retry queue."""
    global _retry_queue
    if _retry_queue:
        _retry_queue.stop()
        _retry_queue = None
