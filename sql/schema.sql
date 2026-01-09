-- ============================================================================
-- BiyometrikFoto.tr Database Schema
-- Supabase PostgreSQL
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- JOBS TABLE
-- Core table for tracking photo analysis jobs
-- ============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Status: PENDING -> PROCESSING -> ANALYZED -> (PASS|WARN|FAIL)
    status TEXT NOT NULL DEFAULT 'PENDING',
    
    -- Timestamps (always UTC)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Image paths (local filesystem or future storage keys)
    original_image_path TEXT,
    normalized_image_path TEXT,
    processed_image_path TEXT,
    
    -- Analysis results stored as JSONB for flexibility
    -- Contains: issues[], metrics, debug_info, etc.
    analysis_result JSONB,
    
    -- Image metadata
    final_image_mime TEXT DEFAULT 'image/jpeg',
    
    -- Acknowledgement tracking for WARN issues
    requires_ack_ids JSONB DEFAULT '[]'::jsonb,
    acknowledged_issue_ids JSONB DEFAULT '[]'::jsonb,
    
    -- Computed decision (can user proceed?)
    can_continue BOOLEAN DEFAULT FALSE,
    
    -- Payment state: ANALYZED -> PAYMENT_PENDING -> PAID -> DELIVERED
    payment_state TEXT DEFAULT 'ANALYZED',
    
    -- Optional user email for delivery
    user_email TEXT
);

-- Jobs indexes
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_payment_state ON jobs(payment_state);

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_jobs_updated_at ON jobs;
CREATE TRIGGER update_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PAYMENTS TABLE
-- Tracks payment transactions (Stripe, future: iyzico)
-- ============================================================================
CREATE TABLE IF NOT EXISTS payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign key to jobs
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Payment provider
    provider TEXT NOT NULL DEFAULT 'stripe',
    
    -- Status: PENDING -> PAID | FAILED | CANCELED
    status TEXT NOT NULL DEFAULT 'PENDING',
    
    -- Amount in kuruş (1 TRY = 100 kuruş)
    amount_kurus INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'TRY',
    
    -- Provider reference (checkout session id, payment intent, etc.)
    provider_ref TEXT,
    
    -- Product type: digital | digital_print
    product_type TEXT DEFAULT 'digital',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Webhook idempotency - prevent double processing
    webhook_processed_at TIMESTAMPTZ
);

-- Payments indexes
CREATE INDEX IF NOT EXISTS idx_payments_job_id ON payments(job_id);
CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status);
CREATE INDEX IF NOT EXISTS idx_payments_provider_ref ON payments(provider_ref);

DROP TRIGGER IF EXISTS update_payments_updated_at ON payments;
CREATE TRIGGER update_payments_updated_at
    BEFORE UPDATE ON payments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PRINT_ORDERS TABLE
-- Tracks physical print orders with shipping info
-- ============================================================================
CREATE TABLE IF NOT EXISTS print_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign key to jobs
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Customer info
    full_name TEXT NOT NULL,
    phone TEXT,
    email TEXT,
    
    -- Shipping address
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    district TEXT,
    postal_code TEXT,
    
    -- Order status: PENDING -> PROCESSING -> SHIPPED -> DELIVERED | CANCELED
    status TEXT NOT NULL DEFAULT 'PENDING',
    
    -- Shipping tracking
    tracking_number TEXT,
    shipped_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Print orders indexes
CREATE INDEX IF NOT EXISTS idx_print_orders_job_id ON print_orders(job_id);
CREATE INDEX IF NOT EXISTS idx_print_orders_status ON print_orders(status);

DROP TRIGGER IF EXISTS update_print_orders_updated_at ON print_orders;
CREATE TRIGGER update_print_orders_updated_at
    BEFORE UPDATE ON print_orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VERIFICATION QUERY (run after schema apply)
-- ============================================================================
-- SELECT 
--     table_name, 
--     (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
-- FROM information_schema.tables t
-- WHERE table_schema = 'public' 
-- AND table_name IN ('jobs', 'payments', 'print_orders');
