#!/usr/bin/env python3
"""
Apply database schema to Supabase PostgreSQL.
Idempotent - safe to run multiple times.

Usage:
    python3 scripts/apply_schema.py
"""

import os
import sys
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def normalize_database_url(url: str) -> str:
    """Normalize DATABASE_URL to use postgresql+asyncpg:// scheme."""
    if not url:
        raise ValueError("DATABASE_URL is empty")
    return re.sub(r'^postgres(ql)?://', 'postgresql+asyncpg://', url)


def split_sql_statements(sql: str) -> list:
    """
    Split SQL into individual statements, handling function bodies with $$ delimiters.
    """
    statements = []
    current_stmt = []
    in_function = False
    
    for line in sql.split('\n'):
        stripped = line.strip()
        
        # Track function body start/end (using $$ delimiter)
        dollar_count = line.count('$$')
        if dollar_count == 1:
            in_function = not in_function
        elif dollar_count == 2:
            # Both open and close on same line (like $$ ... $$ inline)
            pass  # in_function stays the same
        
        current_stmt.append(line)
        
        # Statement ends with semicolon and we're not inside a function body
        if stripped.endswith(';') and not in_function:
            stmt_text = '\n'.join(current_stmt).strip()
            # Skip empty or comment-only statements
            clean_lines = [l.strip() for l in stmt_text.split('\n') 
                          if l.strip() and not l.strip().startswith('--')]
            if clean_lines:
                statements.append(stmt_text)
            current_stmt = []
    
    return statements


async def apply_schema():
    """Apply schema.sql to database."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Mask URL for logging
    masked_url = re.sub(r'://[^@]+@', '://***:***@', database_url)
    print(f"📦 Connecting to: {masked_url[:60]}...")
    
    # Read schema file
    schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)
    
    schema_sql = schema_path.read_text()
    print(f"📄 Loaded schema: {len(schema_sql)} bytes")
    
    # Split into statements
    statements = split_sql_statements(schema_sql)
    print(f"📝 Found {len(statements)} SQL statements")
    
    # Create engine with isolation_level for DDL
    try:
        normalized_url = normalize_database_url(database_url)
        engine = create_async_engine(
            normalized_url, 
            echo=False,
            isolation_level="AUTOCOMMIT"  # Required for DDL statements
        )
    except Exception as e:
        print(f"❌ Failed to create engine: {e}")
        sys.exit(1)
    
    # Apply schema - each statement in its own transaction
    try:
        executed = 0
        skipped = 0
        errors = 0
        
        async with engine.connect() as conn:
            for i, stmt in enumerate(statements, 1):
                # Get first line for logging
                first_line = stmt.split('\n')[0][:60]
                
                try:
                    await conn.execute(text(stmt))
                    executed += 1
                    print(f"   ✅ [{i}/{len(statements)}] {first_line}...")
                except Exception as e:
                    error_str = str(e).lower()
                    if 'already exists' in error_str or 'duplicate' in error_str:
                        skipped += 1
                        print(f"   ⏭️  [{i}/{len(statements)}] Already exists: {first_line}...")
                    else:
                        errors += 1
                        # Truncate error for readability
                        err_msg = str(e)[:150].replace('\n', ' ')
                        print(f"   ❌ [{i}/{len(statements)}] Error: {err_msg}")
        
        print(f"\n📊 Results: {executed} executed, {skipped} skipped, {errors} errors")
        
        # Verify tables exist
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('jobs', 'payments', 'print_orders')
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
            print(f"📊 Tables verified: {', '.join(tables) if tables else 'none'}")
            
            if len(tables) != 3:
                print(f"⚠️  Expected 3 tables, found {len(tables)}")
            else:
                # Count columns per table
                for table in tables:
                    result = await conn.execute(text(f"""
                        SELECT COUNT(*) 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}';
                    """))
                    count = result.scalar()
                    print(f"   - {table}: {count} columns")
        
        await engine.dispose()
        
        if errors == 0:
            print("\n🎉 Schema deployment complete!")
        else:
            print(f"\n⚠️  Schema deployment completed with {errors} errors")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to apply schema: {e}")
        await engine.dispose()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(apply_schema())
