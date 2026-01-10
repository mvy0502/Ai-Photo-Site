"""
Database Connection Pool for Background Workers

Provides a shared connection pool for synchronous DB operations
in background threads (avoiding per-request connection overhead).
"""

import os
import re
import threading
from typing import Optional, Tuple, Any
from contextlib import contextmanager

try:
    from psycopg2 import pool as pg_pool
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class DBPool:
    """
    Thread-safe connection pool for psycopg2.
    
    Usage:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    
    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 5):
        self._dsn = dsn
        self._pool: Optional[pg_pool.ThreadedConnectionPool] = None
        self._lock = threading.Lock()
        self._minconn = minconn
        self._maxconn = maxconn
        self._initialized = False
    
    def initialize(self) -> Tuple[bool, str]:
        """Initialize the connection pool."""
        if not PSYCOPG2_AVAILABLE:
            return False, "psycopg2 not installed"
        
        with self._lock:
            if self._initialized:
                return True, "Already initialized"
            
            try:
                self._pool = pg_pool.ThreadedConnectionPool(
                    self._minconn,
                    self._maxconn,
                    self._dsn,
                    # Connection options
                    connect_timeout=10,
                    options='-c statement_timeout=30000'  # 30s statement timeout
                )
                self._initialized = True
                print(f"✅ [DB_POOL] Initialized (min={self._minconn}, max={self._maxconn})")
                return True, "Pool initialized"
            except Exception as e:
                return False, f"Failed to create pool: {e}"
    
    def is_initialized(self) -> bool:
        return self._initialized and self._pool is not None
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.
        Connection is automatically returned to pool on exit.
        """
        if not self._pool:
            raise RuntimeError("DB pool not initialized")
        
        conn = None
        try:
            conn = self._pool.getconn()
            conn.autocommit = False
            yield conn
        finally:
            if conn:
                try:
                    # Rollback any uncommitted transaction
                    conn.rollback()
                except:
                    pass
                self._pool.putconn(conn)
    
    def execute_with_retry(
        self, 
        query: str, 
        params: tuple = None,
        max_retries: int = 2
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute a query with automatic retry on connection errors.
        
        Returns:
            (success, result, error_message)
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        
                        # Try to fetch results
                        try:
                            result = cur.fetchall()
                        except psycopg2.ProgrammingError:
                            # No results (INSERT/UPDATE)
                            result = cur.rowcount
                        
                        conn.commit()
                        return True, result, None
                        
            except psycopg2.OperationalError as e:
                last_error = str(e)
                if attempt < max_retries:
                    print(f"⚠️ [DB_POOL] Retry {attempt + 1}/{max_retries}: {last_error[:50]}")
                    continue
                    
            except Exception as e:
                last_error = str(e)
                break
        
        return False, None, last_error
    
    def close(self):
        """Close all connections in the pool."""
        with self._lock:
            if self._pool:
                try:
                    self._pool.closeall()
                except:
                    pass
                self._pool = None
                self._initialized = False
                print("🔌 [DB_POOL] Closed")


# Global singleton
_db_pool: Optional[DBPool] = None
_pool_lock = threading.Lock()


def get_db_pool() -> Optional[DBPool]:
    """Get or create the global DB pool."""
    global _db_pool
    
    if _db_pool and _db_pool.is_initialized():
        return _db_pool
    
    with _pool_lock:
        # Double-check after acquiring lock
        if _db_pool and _db_pool.is_initialized():
            return _db_pool
        
        # Get DATABASE_URL
        database_url = os.environ.get("DATABASE_URL", "").strip()
        if not database_url:
            print("⚠️ [DB_POOL] DATABASE_URL not set")
            return None
        
        # Convert postgres:// to postgresql:// for psycopg2
        dsn = re.sub(r'^postgres://', 'postgresql://', database_url)
        
        # Create and initialize pool
        _db_pool = DBPool(dsn, minconn=1, maxconn=3)
        success, msg = _db_pool.initialize()
        
        if not success:
            print(f"❌ [DB_POOL] {msg}")
            _db_pool = None
            return None
        
        return _db_pool


def close_db_pool():
    """Close the global DB pool."""
    global _db_pool
    if _db_pool:
        _db_pool.close()
        _db_pool = None
