"""
Simple Rate Limiter for Status Polling

Prevents aggressive polling from overloading the server.
Uses in-memory storage (suitable for single-instance deployment).
"""

import time
import threading
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from functools import wraps


@dataclass
class RateLimitEntry:
    last_request: float
    request_count: int
    window_start: float


class RateLimiter:
    """
    Token bucket rate limiter for status endpoint.
    
    Default: 1 request per second per job_id
    """
    
    def __init__(
        self,
        requests_per_second: float = 1.0,
        burst_size: int = 3,
        cleanup_interval: int = 60
    ):
        self._requests_per_second = requests_per_second
        self._burst_size = burst_size
        self._min_interval = 1.0 / requests_per_second
        self._entries: Dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval
    
    def check(self, key: str) -> Tuple[bool, Optional[float]]:
        """
        Check if request is allowed.
        
        Returns:
            (allowed, retry_after)
            - allowed: True if request is allowed
            - retry_after: Seconds to wait if not allowed (None if allowed)
        """
        now = time.time()
        
        with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup(now)
            
            entry = self._entries.get(key)
            
            if entry is None:
                # First request for this key
                self._entries[key] = RateLimitEntry(
                    last_request=now,
                    request_count=1,
                    window_start=now
                )
                return True, None
            
            time_since_last = now - entry.last_request
            
            if time_since_last >= self._min_interval:
                # Enough time has passed
                entry.last_request = now
                entry.request_count += 1
                return True, None
            else:
                # Too soon - calculate retry-after
                retry_after = self._min_interval - time_since_last
                return False, retry_after
    
    def _cleanup(self, now: float):
        """Remove stale entries."""
        stale_threshold = now - 60  # Remove entries older than 60s
        
        to_remove = [
            key for key, entry in self._entries.items()
            if entry.last_request < stale_threshold
        ]
        
        for key in to_remove:
            del self._entries[key]
        
        self._last_cleanup = now


# Response cache for status endpoint
class StatusCache:
    """
    Simple cache for status responses.
    Caches responses for 1 second to reduce DB load.
    """
    
    def __init__(self, ttl_seconds: float = 1.0):
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[dict]:
        """Get cached value if not expired."""
        now = time.time()
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            cached_at, value = entry
            if now - cached_at > self._ttl:
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: dict):
        """Cache a value."""
        now = time.time()
        
        with self._lock:
            self._cache[key] = (now, value)
    
    def invalidate(self, key: str):
        """Remove a key from cache."""
        with self._lock:
            self._cache.pop(key, None)


# Global instances
_rate_limiter: Optional[RateLimiter] = None
_status_cache: Optional[StatusCache] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_status_cache() -> StatusCache:
    """Get the global status cache."""
    global _status_cache
    if _status_cache is None:
        _status_cache = StatusCache()
    return _status_cache
