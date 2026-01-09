#!/usr/bin/env python3
"""
Cleanup old jobs from database and optionally delete associated files.

Environment variables:
    JOB_TTL_DAYS: Number of days after which jobs are deleted (default: 7)
    DRY_RUN: If "true", only show what would be deleted (default: true)

Usage:
    # Dry run (default)
    python3 scripts/cleanup_jobs.py
    
    # Actually delete
    DRY_RUN=false python3 scripts/cleanup_jobs.py
    
    # Custom TTL
    JOB_TTL_DAYS=30 python3 scripts/cleanup_jobs.py
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


# Configuration
JOB_TTL_DAYS = int(os.getenv("JOB_TTL_DAYS", "7"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"


def normalize_database_url(url: str) -> str:
    """Normalize DATABASE_URL to use postgresql+asyncpg:// scheme."""
    if not url:
        raise ValueError("DATABASE_URL is empty")
    return re.sub(r'^postgres(ql)?://', 'postgresql+asyncpg://', url)


async def cleanup_jobs():
    """Delete old jobs and their files."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Mask URL for logging
    masked_url = re.sub(r'://[^@]+@', '://***:***@', database_url)
    print(f"📦 Connecting to: {masked_url[:60]}...")
    print(f"⚙️  Config: TTL={JOB_TTL_DAYS} days, DRY_RUN={DRY_RUN}")
    print()
    
    # Create engine
    try:
        normalized_url = normalize_database_url(database_url)
        engine = create_async_engine(normalized_url, echo=False)
    except Exception as e:
        print(f"❌ Failed to create engine: {e}")
        sys.exit(1)
    
    try:
        async with engine.connect() as conn:
            # Get jobs to delete
            result = await conn.execute(text(f"""
                SELECT id, original_image_path, processed_image_path, created_at
                FROM jobs
                WHERE created_at < NOW() - INTERVAL '{JOB_TTL_DAYS} days'
                ORDER BY created_at ASC
            """))
            rows = result.fetchall()
            
            if not rows:
                print(f"✅ No jobs older than {JOB_TTL_DAYS} days found.")
                await engine.dispose()
                return
            
            print(f"📋 Found {len(rows)} jobs to {'review' if DRY_RUN else 'delete'}:")
            print()
            
            files_to_delete = []
            
            for row in rows:
                job_id = str(row[0])
                original_path = row[1]
                processed_path = row[2]
                created_at = row[3]
                
                age_days = (datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc)).days
                
                print(f"  📁 Job: {job_id[:8]}...")
                print(f"     Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')} ({age_days} days ago)")
                
                # Check for associated files
                if original_path:
                    original_file = Path(original_path)
                    if original_file.exists():
                        files_to_delete.append(original_file)
                        print(f"     Original: {original_path} (exists)")
                    else:
                        print(f"     Original: {original_path} (not found)")
                
                if processed_path:
                    # processed_path is usually a URL like /uploads/xxx_processed.png
                    # Convert to filesystem path
                    if processed_path.startswith("/uploads/"):
                        processed_file = Path(processed_path.replace("/uploads/", "uploads/"))
                    else:
                        processed_file = Path(processed_path)
                    
                    if processed_file.exists():
                        files_to_delete.append(processed_file)
                        print(f"     Processed: {processed_file} (exists)")
                    else:
                        print(f"     Processed: {processed_file} (not found)")
                
                print()
            
            # Summary
            print(f"📊 Summary:")
            print(f"   Jobs to delete: {len(rows)}")
            print(f"   Files to delete: {len(files_to_delete)}")
            print()
            
            if DRY_RUN:
                print("🔍 DRY RUN - No changes made.")
                print("   Set DRY_RUN=false to actually delete.")
            else:
                # Delete files first
                files_deleted = 0
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        files_deleted += 1
                    except Exception as e:
                        print(f"   ⚠️ Failed to delete {file_path}: {e}")
                
                # Delete DB records
                async with engine.begin() as tx_conn:
                    result = await tx_conn.execute(text(f"""
                        DELETE FROM jobs
                        WHERE created_at < NOW() - INTERVAL '{JOB_TTL_DAYS} days'
                    """))
                    deleted_count = result.rowcount
                
                print(f"✅ Deleted {deleted_count} jobs and {files_deleted} files.")
        
        await engine.dispose()
        print("\n🎉 Cleanup complete!")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        await engine.dispose()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    
    print("=" * 60)
    print("🧹 BiyometrikFoto.tr Job Cleanup")
    print("=" * 60)
    print()
    
    asyncio.run(cleanup_jobs())
