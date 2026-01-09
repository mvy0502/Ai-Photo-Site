# 🇹🇷 BiyometrikFoto.tr

Türkiye standartlarına uygun biyometrik fotoğraf hazırlama servisi. Pasaport, vize ve resmi belgeler için kabul garantili fotoğraf.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Özellikler

- 🤖 **AI Analiz** - MediaPipe ile yüz tespiti ve biyometrik kontroller
- 🖼️ **Arka Plan Kaldırma** - PhotoRoom API ile profesyonel beyaz arka plan
- 📐 **Türkiye Standartları** - 50×60mm, 300 DPI, ICAO uyumlu
- 💳 **Ödeme Entegrasyonu** - Stripe ile güvenli ödeme (opsiyonel)
- 📧 **E-posta Gönderimi** - İndirme linki e-posta ile
- 🗄️ **Veritabanı** - Supabase PostgreSQL ile kalıcı depolama

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- PostgreSQL (Supabase önerilir)
- PhotoRoom API anahtarı

### Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/mvy0502/Ai-Photo-Site.git
cd Ai-Photo-Site

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt

# Environment variables
cp CONFIG_TEMPLATE.md .env
# .env dosyasını düzenle

# Veritabanı şeması
python scripts/apply_schema.py

# Sunucuyu başlat
uvicorn app:app --reload
```

### Environment Variables

```env
# Zorunlu
DATABASE_URL=postgresql://...
PHOTOROOM_API_KEY=sk_...

# Opsiyonel (Ödeme)
STRIPE_SECRET_KEY=sk_...
STRIPE_PUBLISHABLE_KEY=pk_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Opsiyonel (E-posta)
SMTP_HOST=smtp.gmail.com
SMTP_USER=...
SMTP_PASS=...
EMAIL_FROM=...
```

## 📁 Proje Yapısı

```
├── app.py                 # Ana FastAPI uygulaması
├── requirements.txt       # Python bağımlılıkları
├── render.yaml           # Render deployment config
├── DEPLOY.md             # Deployment rehberi
├── CONFIG_TEMPLATE.md    # Environment template
│
├── utils/                # Yardımcı modüller
│   ├── analyze_v2.py     # V2 biyometrik analiz
│   ├── photoroom_client.py # PhotoRoom API
│   ├── db.py             # Veritabanı bağlantısı
│   ├── db_jobs.py        # Job CRUD işlemleri
│   ├── payment.py        # Stripe entegrasyonu
│   └── email_service.py  # E-posta servisi
│
├── static/               # Frontend dosyaları
│   ├── app.js           # JavaScript
│   ├── styles.css       # CSS
│   └── images/          # Görseller
│
├── templates/            # Jinja2 templates
│   ├── index.html       # Ana sayfa
│   ├── payment_success.html
│   └── payment_cancel.html
│
├── sql/                  # Veritabanı
│   └── schema.sql       # Tablo tanımları
│
├── scripts/              # Yardımcı scriptler
│   ├── apply_schema.py  # Şema uygulama
│   └── cleanup_jobs.py  # Eski job temizliği
│
├── models/               # ML modelleri
│   ├── face_landmarker.task
│   └── selfie_segmenter.tflite
│
└── tests/                # Test dosyaları
```

## 🔧 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | Ana sayfa |
| `/upload` | POST | Fotoğraf yükleme |
| `/job/{id}/status` | GET | İş durumu |
| `/process/{id}` | POST | PhotoRoom işleme |
| `/api/download/{id}` | GET | Güvenli indirme |
| `/api/health` | GET | Sağlık kontrolü |
| `/api/health/db` | GET | DB sağlık kontrolü |

## 🌐 Deployment

Detaylı deployment rehberi için: [DEPLOY.md](DEPLOY.md)

### Render (Önerilen)

```bash
# render.yaml otomatik algılanır
# Dashboard'dan environment variables ekle
```

### Docker (Yakında)

```bash
docker build -t biyometrikfoto .
docker run -p 8000:8000 biyometrikfoto
```

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing`)
5. Pull Request açın

---

**BiyometrikFoto.tr** - Türkiye'nin biyometrik fotoğraf servisi 🇹🇷
