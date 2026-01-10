"""
Supabase PostgreSQL Database Connection
Uses async SQLAlchemy + asyncpg

PgBouncer Compatibility:
- Disables asyncpg prepared statement cache (critical for transaction pooling)
- Uses conservative pool sizes for Render free tier
- Supports env toggles for fine-tuning

Environment Variables:
- DATABASE_URL: Required connection string
- DB_DISABLE_PREPARED_STATEMENTS: Set to "true" to disable asyncpg statement cache (recommended for PgBouncer)
- DB_POOL_SIZE: Connection pool size (default: 2)
- DB_MAX_OVERFLOW: Max overflow connections (default: 3)
- DB_POOL_PRE_PING: Enable connection health checks (default: true)
"""

import os
import re
from typing import AsyncGenerator, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text
from sqlalchemy.pool import NullPool

# Import sanitized env helpers
from utils.env_config import get_env, get_database_url, validate_database_url, mask_url_credentials

load_dotenv()

# Environment variable names (for documentation)
REQUIRED_ENV_VARS = ["DATABASE_URL"]

# PgBouncer / Pooler configuration
def _get_pool_config() -> dict:
    """Get database pool configuration from environment."""
    disable_prepared = get_env("DB_DISABLE_PREPARED_STATEMENTS", default="true").lower() == "true"
    pool_size = int(get_env("DB_POOL_SIZE", default="2"))
    max_overflow = int(get_env("DB_MAX_OVERFLOW", default="3"))
    pool_pre_ping = get_env("DB_POOL_PRE_PING", default="true").lower() == "true"
    
    return {
        "disable_prepared_statements": disable_prepared,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_pre_ping": pool_pre_ping
    }

def _get_connect_args(disable_prepared: bool) -> dict:
    """
    Get asyncpg connection arguments.
    
    Critical for PgBouncer: Disable prepared statement cache to avoid
    'prepared statement X already exists' errors in transaction pooling mode.
    """
    if disable_prepared:
        return {
            "statement_cache_size": 0,  # Disable prepared statement cache
            "prepared_statement_cache_size": 0,  # Redundant but explicit
        }
    return {}


def normalize_database_url(url: str) -> str:
    """
    Normalize DATABASE_URL to use postgresql+asyncpg:// scheme.
    Supabase typically provides postgres:// or postgresql://
    """
    if not url:
        raise ValueError("DATABASE_URL is empty")
    
    # Replace postgres:// or postgresql:// with postgresql+asyncpg://
    normalized = re.sub(r'^postgres(ql)?://', 'postgresql+asyncpg://', url)
    return normalized


def get_pooler_url(url: str) -> Optional[str]:
    """
    Try to construct a session pooler URL (port 6543) from the direct connection URL.
    Only works if the host contains 'supabase' patterns.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        
        # Check if it's a Supabase URL
        if 'supabase' not in host.lower():
            return None
        
        # If already using pooler port, return None
        if parsed.port == 6543:
            return None
        
        # Try to construct pooler URL
        # Supabase pooler format: aws-0-[region].pooler.supabase.com:6543
        # Direct format: db.[project-ref].supabase.co:5432
        
        # If it's already a pooler URL, just change the port
        if 'pooler.supabase.com' in host:
            new_netloc = f"{parsed.username}:{parsed.password}@{host}:6543" if parsed.password else f"{host}:6543"
            return urlunparse((parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
        
        # For direct connection URLs, we can't reliably construct pooler URL
        # User should provide POOLER_DATABASE_URL separately if needed
        return None
    except Exception:
        return None


class DatabaseManager:
    """Manages async database connections with fallback support."""
    
    def __init__(self):
        self.engine = None
        self.async_session_factory = None
        self.connection_info = {"url_type": "unknown", "port": None, "connected": False}
    
    async def initialize(self) -> Tuple[bool, str]:
        """
        Initialize database connection.
        Returns (success: bool, message: str)
        """
        # Get sanitized DATABASE_URL (handles newlines, whitespace, etc.)
        database_url, warnings = get_database_url(required=False)
        
        # Log any warnings
        for warning in warnings:
            print(f"⚠️ [DB CONFIG] {warning}")
        
        if not database_url:
            return False, "DATABASE_URL environment variable is not set"
        
        # Get pool configuration
        pool_config = _get_pool_config()
        connect_args = _get_connect_args(pool_config["disable_prepared_statements"])
        
        # Log configuration (non-secret)
        print(f"🔧 [DB CONFIG] Pool size: {pool_config['pool_size']}, "
              f"max_overflow: {pool_config['max_overflow']}, "
              f"pre_ping: {pool_config['pool_pre_ping']}, "
              f"disable_prepared_statements: {pool_config['disable_prepared_statements']}")
        
        # Try primary connection
        try:
            normalized_url = normalize_database_url(database_url)
            parsed = urlparse(normalized_url)
            
            self.engine = create_async_engine(
                normalized_url,
                pool_pre_ping=pool_config["pool_pre_ping"],
                pool_size=pool_config["pool_size"],
                max_overflow=pool_config["max_overflow"],
                connect_args=connect_args,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            url_type = "pooler" if parsed.port == 6543 or "pooler" in (parsed.hostname or "") else "direct"
            self.connection_info = {
                "url_type": url_type,
                "port": parsed.port or 5432,
                "connected": True,
                "prepared_statements_disabled": pool_config["disable_prepared_statements"]
            }
            
            return True, f"Connected via {url_type} (port {self.connection_info['port']}, prepared_stmts={not pool_config['disable_prepared_statements']})"
            
        except Exception as e:
            error_msg = str(e)
            # Sanitize error message to not expose secrets
            sanitized_error = mask_url_credentials(error_msg)
            
            # Try pooler fallback if available
            pooler_url = get_pooler_url(database_url)
            if pooler_url:
                try:
                    normalized_pooler = normalize_database_url(pooler_url)
                    self.engine = create_async_engine(
                        normalized_pooler,
                        pool_pre_ping=pool_config["pool_pre_ping"],
                        pool_size=pool_config["pool_size"],
                        max_overflow=pool_config["max_overflow"],
                        connect_args=connect_args,
                        echo=False
                    )
                    
                    async with self.engine.connect() as conn:
                        result = await conn.execute(text("SELECT 1"))
                        result.fetchone()
                    
                    self.async_session_factory = async_sessionmaker(
                        self.engine,
                        class_=AsyncSession,
                        expire_on_commit=False
                    )
                    
                    self.connection_info = {
                        "url_type": "pooler",
                        "port": 6543,
                        "connected": True,
                        "prepared_statements_disabled": pool_config["disable_prepared_statements"]
                    }
                    
                    return True, f"Connected via session pooler (port 6543, prepared_stmts={not pool_config['disable_prepared_statements']})"
                    
                except Exception as pooler_error:
                    pooler_sanitized = mask_url_credentials(str(pooler_error))
                    return False, f"Direct failed: {sanitized_error[:100]}. Pooler also failed: {pooler_sanitized[:100]}"
            
            return False, f"Connection failed: {sanitized_error[:200]}"
    
    async def test_connection(self) -> Tuple[bool, int]:
        """
        Test database connection by running SELECT 1.
        Returns (success: bool, result: int)
        """
        if not self.engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.fetchone()
            return True, row[0] if row else 0
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency for FastAPI endpoints."""
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.async_session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async for session in db_manager.get_session():
        yield session
