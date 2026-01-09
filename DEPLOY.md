# 🚀 Render Deployment Guide

## Prerequisites

- GitHub repository connected to Render
- Supabase Postgres database (already configured)
- PhotoRoom API key (production)
- Stripe account with API keys (optional - see Payment section)

---

## Quick Deploy

### Option 1: Blueprint (Recommended)

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New** → **Blueprint**
4. Connect your repository
5. Render will detect `render.yaml` and configure automatically
6. Set environment variables in dashboard (see below)
7. Click **Apply**

### Option 2: Manual Setup

1. **New Web Service** in Render Dashboard
2. Connect GitHub repository
3. Configure:

| Setting | Value |
|---------|-------|
| **Runtime** | Python |
| **Region** | Frankfurt (EU) |
| **Build Command** | `pip install --upgrade pip && pip install -r requirements.txt` |
| **Start Command** | See below |

---

## Start Command

```bash
gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 5 --access-logfile - --error-logfile -
```

### Worker Count Recommendations

| Plan | Workers | Memory |
|------|---------|--------|
| Starter | 2 | 512MB |
| Standard | 4 | 2GB |
| Pro | 8 | 4GB |

---

## Environment Variables

Set these in Render Dashboard → Environment:

### Required

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Supabase Postgres connection string |
| `PHOTOROOM_API_KEY` | PhotoRoom production API key |

### Payment (Optional)

> ⚠️ **Payments are optional.** When Stripe variables are not set, the app runs in "beta mode" with free downloads. Users can download their photos without payment.

| Variable | Description |
|----------|-------------|
| `STRIPE_SECRET_KEY` | Stripe secret key (sk_live_...) |
| `STRIPE_PUBLISHABLE_KEY` | Stripe publishable key (pk_live_...) |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |
| `PAYMENTS_ENABLED` | Set to `false` to disable payments even if Stripe keys exist |

**Payment Behavior:**
- If ALL Stripe vars are set → Payments enabled, checkout required
- If ANY Stripe var is missing → Payments disabled, free downloads
- If `PAYMENTS_ENABLED=false` → Payments disabled (override)

### Email (Optional)

| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | SMTP server (e.g., smtp.gmail.com) |
| `SMTP_USER` | SMTP username |
| `SMTP_PASS` | SMTP password / app password |
| `EMAIL_FROM` | Sender email address |

### App Config (Defaults set in render.yaml)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEV_ALLOW_MEMORY_FALLBACK` | `false` | Memory fallback for DB errors |
| `JOB_TTL_DAYS` | `7` | Days before old jobs are deleted |
| `PAYMENTS_ENABLED` | auto | Set to `false` to force disable payments |

---

## Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/health` | Basic health check (used by Render) |
| `/api/health/db` | Database connectivity check |

---

## Post-Deployment Verification

### 1. Basic Health Check
```bash
curl https://your-app.onrender.com/api/health
# Expected: {"status": "ok", "service": "biyometrikfoto-api", "version": "..."}
```

### 2. Database Check
```bash
curl https://your-app.onrender.com/api/health/db
# Expected: {"ok": true, "db": 1, "connection_info": {...}}
```

### 3. Homepage Loads
```bash
curl -I https://your-app.onrender.com/
# Expected: HTTP/2 200
```

### 4. Upload Test
```bash
curl -X POST -F "photo=@test.jpg" https://your-app.onrender.com/upload
# Expected: {"success": true, "job_id": "..."}
```

---

## Common Issues

### 1. Port Binding Error
- ❌ `--bind 0.0.0.0:8000` (hardcoded port)
- ✅ `--bind 0.0.0.0:$PORT` (Render provides PORT)

### 2. Database Connection Failed
- Check `DATABASE_URL` format: `postgresql://user:pass@host:port/db`
- Ensure Supabase allows connections from Render IPs
- Try Session Pooler (port 6543) if direct fails

### 3. Static Files Not Loading
- Verify `app.mount("/static", ...)` is present
- Check `uploads/` directory exists (created on first upload)

### 4. Timeout on Image Processing
- Default timeout: 120s (sufficient for PhotoRoom)
- Increase if needed: `--timeout 180`

### 5. Memory Issues
- MediaPipe + OpenCV use ~300MB
- Use `opencv-python-headless` (no GUI dependencies)
- Upgrade to Standard plan if Starter OOMs

---

## Stripe Webhook Setup

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://your-app.onrender.com/api/webhook/stripe`
3. Select events:
   - `checkout.session.completed`
   - `payment_intent.succeeded`
4. Copy signing secret → Set as `STRIPE_WEBHOOK_SECRET`

---

## Monitoring

### Logs
- Render Dashboard → Your Service → Logs
- Filter by: `ERROR`, `WARNING`, `[DB]`, `[PHOTOROOM]`

### Metrics
- Render Dashboard → Your Service → Metrics
- Watch: CPU, Memory, Response Time

---

## Scaling

### Horizontal (more workers)
```bash
gunicorn app:app --workers 4 ...
```

### Vertical (bigger instance)
- Upgrade plan in Render Dashboard

---

## Rollback

1. Render Dashboard → Your Service → Events
2. Find previous successful deploy
3. Click **Rollback**

---

## Local Production Test

```bash
# Install gunicorn
pip install gunicorn

# Run locally with production settings
PORT=8000 gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
```
