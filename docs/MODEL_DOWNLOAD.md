# Model Dosyası İndirme Rehberi

## Hızlı Başlangıç

### Yöntem 1: Python Script (Önerilen - Progress gösterir)

```bash
python3 download_model.py
```

Bu script:
- Progress bar gösterir
- Dosya zaten varsa uyarır
- Hata durumunda temizlik yapar

### Yöntem 2: Shell Script

```bash
chmod +x download_model.sh
./download_model.sh
```

### Yöntem 3: Manuel İndirme

```bash
# Klasörü oluştur
mkdir -p models

# Model dosyasını indir (blendshapes versiyonu - önerilen)
curl -L -o models/face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_blendshapes/float16/1/face_landmarker_v2_with_blendshapes.task
```

## Model Dosyası Doğrulama

İndirme tamamlandıktan sonra kontrol edin:

```bash
ls -lh models/
```

**Beklenen çıktı:**
```
face_landmarker_v2_with_blendshapes.task  9.2M  (yaklaşık 9-10 MB)
```

## Alternatif Model (Blendshapes olmadan)

Eğer blendshapes versiyonu çalışmazsa, basic model'i deneyin:

```bash
python3 download_model.py basic
```

veya

```bash
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Sorun Giderme

### İndirme başarısız olursa:
1. İnternet bağlantınızı kontrol edin
2. Firewall/proxy ayarlarınızı kontrol edin
3. Manuel olarak tarayıcıdan indirip `models/` klasörüne koyun

### Model dosyası bozuksa:
```bash
rm models/face_landmarker_v2_with_blendshapes.task
python3 download_model.py
```

## Model Dosyası Boyutu

- **face_landmarker_v2_with_blendshapes.task**: ~9-10 MB
- **face_landmarker.task**: ~8-9 MB

İndirme süresi internet hızınıza bağlı olarak 10 saniye - 2 dakika arası değişebilir.
