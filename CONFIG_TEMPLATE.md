# BiyometrikFoto.tr - Configuration Template

Create a `.env` file in the project root with the following variables:

```bash
# ============================================================================
# Stripe Configuration
# ============================================================================
# Get your keys from: https://dashboard.stripe.com/apikeys
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx

# ============================================================================
# Pricing (in kuruş - 100 TL = 10000 kuruş)
# ============================================================================
DIGITAL_PRICE_KURUS=10000
PRINT_PRICE_KURUS=20000

# ============================================================================
# Download URL Security
# ============================================================================
# Change this to a random secret string in production!
DOWNLOAD_URL_SECRET=your-secret-key-change-in-production

# ============================================================================
# Email Configuration (Optional)
# ============================================================================
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
EMAIL_FROM=noreply@biyometrikfoto.tr
EMAIL_FROM_NAME=BiyometrikFoto.tr
EMAIL_RATE_LIMIT_SECONDS=300

# ============================================================================
# PhotoRoom API (Existing)
# ============================================================================
PHOTOROOM_API_KEY=your-photoroom-api-key
```

## Stripe Webhook Setup

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://yourdomain.com/api/webhook/stripe`
3. Select event: `checkout.session.completed`
4. Copy the webhook signing secret to `STRIPE_WEBHOOK_SECRET`

## Testing Locally with Stripe CLI

```bash
# Install Stripe CLI
brew install stripe/stripe-cli/stripe

# Login
stripe login

# Forward webhooks to local server
stripe listen --forward-to localhost:8000/api/webhook/stripe

# Copy the webhook secret from output to .env
```
