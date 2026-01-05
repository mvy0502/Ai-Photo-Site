# AI Photo Site - Biyometrik Fotoğraf Kontrol Sistemi

Modern, AI destekli biyometrik fotoğraf kontrol ve analiz platformu. FastAPI, OpenCV ve MediaPipe kullanarak fotoğraf kalitesini otomatik olarak değerlendirir.

## ✨ Özellikler

- 📸 **Fotoğraf Yükleme**: JPG, PNG, WEBP formatlarında fotoğraf yükleme
- 🤖 **AI Analiz**: OpenCV ve MediaPipe ile otomatik görüntü analizi
- ✅ **Kalite Kontrolü**: Yüz tespiti, bulanıklık, parlaklık ve kadraj kontrolü
- 🎨 **Modern UI**: PhotoAid benzeri modal akış ve gerçek zamanlı progress gösterimi
- ⚡ **Gerçek Zamanlı İşleme**: Background task'lar ile asenkron analiz
- 📊 **Detaylı Raporlama**: PASS/FAIL sonuçları ve nedenleri

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.8+
- pip

### Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd ai-photo-site
```

2. **Virtual environment oluşturun:**
```bash
python -m venv .venv
```

3. **Virtual environment'ı aktifleştirin:**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

4. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

**Not:** MediaPipe kurulumunda sorun yaşarsanız, sisteminizde gerekli bağımlılıkların yüklü olduğundan emin olun.

### Çalıştırma

```bash
uvicorn app:app --reload
```

Uygulama `http://localhost:8000` adresinde çalışacaktır.

## 📁 Proje Yapısı

```
ai-photo-site/
├── app.py                 # FastAPI ana uygulama
├── requirements.txt        # Python bağımlılıkları
├── README.md             # Bu dosya
├── .gitignore            # Git ignore kuralları
├── templates/            # Jinja2 HTML şablonları
│   ├── index.html        # Ana sayfa
│   ├── job.html          # Job durum sayfası
│   └── ...
├── static/               # Statik dosyalar (CSS, JS)
│   ├── styles.css        # Özel stiller
│   └── app.js            # Frontend JavaScript
└── uploads/              # Yüklenen fotoğraflar (gitignore)
```

## 🎯 Kullanım

1. Tarayıcıda `http://localhost:8000` adresini açın
2. "Başlamadan Önce" butonuna tıklayarak kuralları okuyun
3. Bir fotoğraf seçin ve "Fotoğraf Yükle" butonuna tıklayın
4. Processing ekranında AI analizini izleyin
5. Sonuç ekranında PASS/FAIL durumunu ve detayları görüntüleyin

## 🔍 Analiz Kriterleri

### PASS Kriterleri
- ✅ Tek yüz tespit edildi
- ✅ Yüz net ve odakta
- ✅ Yeterli aydınlatma
- ✅ Uygun kadraj

### FAIL Kriterleri (AI ile düzeltilemez)
- ❌ Yüz tespit edilemedi
- ❌ Birden fazla yüz var
- ❌ Fotoğraf çok bulanık
- ❌ Yüz çok karanlık veya aşırı parlak
- ❌ Yüz kadrajı uygun değil

### Otomatik Düzeltilen (Kullanıcıya gösterilmez)
- 🔧 Arka plan beyazlaştırma
- 🔧 Oran düzeltme (50x60mm)
- 🔧 Işık dengesi
- 🔧 Küçük eğim düzeltmeleri

## 🛠️ Teknolojiler

- **Backend:**
  - FastAPI - Modern Python web framework
  - OpenCV - Görüntü işleme
  - MediaPipe - Yüz tespiti
  - NumPy - Sayısal hesaplamalar

- **Frontend:**
  - HTML5 / CSS3
  - JavaScript (Vanilla)
  - Tailwind CSS - Utility-first CSS framework
  - Jinja2 - Template engine

## 📝 API Endpoints

- `GET /` - Ana sayfa
- `POST /upload` - Fotoğraf yükleme
- `GET /job/{job_id}` - Job durum sayfası
- `GET /job/{job_id}/status` - Job durumu (JSON)
- `GET /uploads` - Yüklenen dosyalar listesi

## 🔧 Yapılandırma

Analiz eşik değerleri `app.py` dosyasında ayarlanabilir:

```python
FACE_BLUR_THRESHOLD = 50.0
FACE_BRIGHTNESS_MIN = 50.0
FACE_BRIGHTNESS_MAX = 240.0
FACE_RATIO_MIN_UNRECOVERABLE = 0.05
FACE_RATIO_MAX_UNRECOVERABLE = 0.60
MIN_RESOLUTION = 400 * 400
```

## 📄 Lisans

Bu proje özel bir projedir.

## 👥 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add some amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📞 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not:** Bu proje geliştirme aşamasındadır. Production kullanımı için ek testler ve optimizasyonlar gerekebilir.
