# 🖼️ AI Görsel Oluşturma Rehberi

Bu dosya, BiyometrikFoto.tr sitesi için gereken tüm görselleri ve AI promptlarını içerir.

**Önerilen Araçlar:** Midjourney, DALL-E 3, Ideogram, Leonardo.ai, Stable Diffusion

---

## ⚠️ EKSİK GÖRSEL - ACİL OLUŞTURULMASI GEREKİYOR

| Görsel | Dosya Yolu | Açıklama |
|--------|------------|----------|
| **Şeffaf Gözlüklü Örnek** | `/static/images/examples/correct-glasses.png` | Şeffaf numaralı gözlük takan kişinin biyometrik fotoğrafı |

**Bu görseli oluşturup belirtilen klasöre kaydedin!**

---

## 📱 1. HERO - Örnek Biyometrik Fotoğraf

**Dosya Adı:** `sample-biometric.jpg`  
**Boyut:** 300 x 360 px (5:6 oran)  
**Kullanım Yeri:** Ana sayfa telefon mockup içinde

### Prompt (Kadın - Önerilen):
```
Professional Turkish biometric passport photo of a young woman aged 25-30,
front facing camera, neutral expression with closed mouth, white background,
even soft lighting with no shadows on face, eyes open and looking directly at camera,
natural makeup, dark hair, official government document photo style,
high resolution, photorealistic, clean and professional
--ar 5:6 --v 6
```

### Prompt (Erkek - Alternatif):
```
Professional Turkish biometric passport photo of a young man aged 25-35,
front facing camera, neutral expression, white background,
even lighting with no shadows, eyes open looking at camera,
clean shaven, dark hair, official passport photo style,
high resolution, photorealistic
--ar 5:6 --v 6
```

---

## 👤 2. DOĞRU ÖRNEKLER (Do's)

**Klasör:** `examples/correct/`  
**Boyut:** 200 x 240 px  
**Kullanım Yeri:** "Doğru Örnekler" bölümü (opsiyonel, ileride eklenebilir)

### correct-woman.jpg
```
Perfect biometric passport photo example, young woman, front view,
neutral expression, white background, proper lighting, 
eyes open, official document photo, green checkmark overlay
--ar 5:6 --v 6
```

### correct-man.jpg
```
Perfect biometric passport photo example, young man, front view,
neutral expression, white background, proper lighting,
eyes open, clean appearance, official style
--ar 5:6 --v 6
```

### correct-hijab.jpg
```
Perfect biometric passport photo example, woman wearing hijab headscarf,
face fully visible, front view, neutral expression, white background,
religious headwear allowed, proper lighting, official document style
--ar 5:6 --v 6
```

### correct-glasses.png ⚠️ OLUŞTURULMASI GEREKİYOR!
**Dosya yolu:** `/static/images/examples/correct-glasses.png`

```
Professional Turkish biometric passport photo of a young professional wearing 
CLEAR TRANSPARENT prescription glasses with thin metal frames. 
Face clearly visible through completely transparent lenses - NOT sunglasses, NOT tinted.
Neutral expression, looking directly at camera, pure white background, 
even soft lighting with no shadows, no glare on lenses, eyes clearly visible.
High quality, official passport photo style, photorealistic.
--ar 5:6 --v 6
```

**Alternatif prompt (daha basit):**
```
Passport photo of Turkish man or woman wearing clear prescription eyeglasses,
transparent lenses, thin frame, neutral expression, white background,
professional biometric ID photo style, eyes visible through glasses
--ar 5:6 --v 6
```

---

## ❌ 3. YANLIŞ ÖRNEKLER (Don'ts)

**Klasör:** `examples/wrong/`  
**Boyut:** 200 x 240 px  
**Kullanım Yeri:** "Yanlış Örnekler" bölümü (opsiyonel)

### wrong-sunglasses.jpg
```
WRONG passport photo example, person wearing dark sunglasses,
eyes not visible, white background, red X mark overlay,
showing incorrect biometric photo, rejection example
--ar 5:6 --v 6
```

### wrong-smiling.jpg
```
WRONG passport photo example, person smiling with teeth showing,
too happy expression, white background, red X mark overlay,
showing what NOT to do for official photo
--ar 5:6 --v 6
```

### wrong-hat.jpg
```
WRONG passport photo example, person wearing a baseball cap,
head covering not allowed, white background, red X mark overlay,
rejection example for biometric photo
--ar 5:6 --v 6
```

### wrong-angle.jpg
```
WRONG passport photo example, person looking to the side,
not facing camera directly, profile angle, white background,
red X mark, showing incorrect pose
--ar 5:6 --v 6
```

---

## 📸 4. NASIL ÇALIŞIR - İllüstrasyonlar

**Klasör:** `steps/`  
**Boyut:** 280 x 200 px  
**Stil:** Flat/minimalist illustration, emerald green (#10b981) renk teması

### step-1-upload.png
```
Minimalist flat illustration, hands holding smartphone taking selfie photo,
clean simple vector style, emerald green and white color scheme,
modern app interface visible on screen, simple geometric shapes,
no text, isolated on white background
--ar 7:5 --v 6 --style raw
```

### step-2-process.png
```
Minimalist flat illustration of AI photo processing,
split screen before and after effect, messy background transforming to white,
digital transformation visualization, emerald green accent color,
simple geometric style, magic wand or sparkle effect
--ar 7:5 --v 6 --style raw
```

### step-3-download.png
```
Minimalist flat illustration of successful download,
document with photo and green checkmark, download arrow icon,
mobile phone showing success screen, emerald green color scheme,
celebration confetti subtle, simple vector style
--ar 7:5 --v 6 --style raw
```

---

## 💡 5. FOTOĞRAF İPUÇLARI - İnfografikler

**Klasör:** `tips/`  
**Boyut:** 200 x 200 px  
**Stil:** Clean infographic style

### tip-distance.png
```
Simple infographic illustration showing correct distance for selfie,
phone 40-50cm away from face, measurement line indicator,
side view silhouette, emerald green color, clean minimal style,
educational diagram, no text needed
--ar 1:1 --v 6 --style raw
```

### tip-position.png
```
Simple infographic showing correct head position for passport photo,
front view face outline with alignment guide lines,
straight shoulders indicator, center position markers,
blue accent color, clean minimal diagram style
--ar 1:1 --v 6 --style raw
```

### tip-lighting.png
```
Simple infographic showing good lighting for photos,
sun/window icon with light rays toward face,
no shadows diagram, face outline with even lighting,
yellow/amber accent color, clean minimal style
--ar 1:1 --v 6 --style raw
```

---

## 🏆 6. GÜVEN ROZETLERİ (Trust Badges)

**Klasör:** `badges/`  
**Boyut:** 120 x 40 px (veya 40 x 40 px kare ikonlar)  
**Format:** PNG transparent background

### badge-ssl.png
```
Simple SSL security badge icon, padlock with checkmark,
emerald green color, minimal flat design, transparent background
--ar 3:1 --v 6
```

### badge-guarantee.png
```
Simple guarantee shield badge icon, shield with checkmark,
emerald green color, minimal flat design, transparent background
--ar 3:1 --v 6
```

### badge-fast.png
```
Simple speed/fast badge icon, clock with lightning bolt,
emerald green color, minimal flat design, transparent background
--ar 3:1 --v 6
```

---

## 📋 Görsel Checklist

| # | Görsel | Dosya Adı | Boyut | Öncelik |
|---|--------|-----------|-------|---------|
| 1 | Hero örnek fotoğraf | `sample-biometric.jpg` | 300x360 | ⭐ Yüksek |
| 2 | Adım 1 - Upload | `steps/step-1-upload.png` | 280x200 | ⭐ Yüksek |
| 3 | Adım 2 - İşlem | `steps/step-2-process.png` | 280x200 | ⭐ Yüksek |
| 4 | Adım 3 - İndir | `steps/step-3-download.png` | 280x200 | ⭐ Yüksek |
| 5 | Mesafe ipucu | `tips/tip-distance.png` | 200x200 | Orta |
| 6 | Pozisyon ipucu | `tips/tip-position.png` | 200x200 | Orta |
| 7 | Aydınlatma ipucu | `tips/tip-lighting.png` | 200x200 | Orta |
| 8 | Doğru örnek (kadın) | `examples/correct-woman.jpg` | 200x240 | Düşük |
| 9 | Yanlış örnek (gözlük) | `examples/wrong-sunglasses.jpg` | 200x240 | Düşük |

---

## 🎨 Renk Paleti

| Renk | Hex | Kullanım |
|------|-----|----------|
| Emerald (Ana) | `#10b981` | Başarı, CTA, vurgu |
| Emerald Dark | `#059669` | Hover durumları |
| Blue | `#3b82f6` | İkincil vurgu |
| Amber | `#f59e0b` | Uyarı, yıldızlar |
| Red | `#ef4444` | Hata, yanlış örnekler |
| Gray | `#6b7280` | Metin, kenarlıklar |

---

## 📝 Notlar

1. **Yüz çeşitliliği:** Farklı ten renkleri ve yaş grupları kullanın
2. **Gerçekçilik:** Photorealistic stil tercih edin, çok karikatürize olmaktan kaçının
3. **Tutarlılık:** Tüm görsellerde aynı renk paletini kullanın
4. **Format:** 
   - Fotoğraflar için `.jpg` (kaliteli sıkıştırma)
   - İllüstrasyonlar için `.png` (şeffaf arka plan)
5. **Optimizasyon:** TinyPNG ile sıkıştırın (web performansı için)

---

## 🚀 Hızlı Başlangıç

En önemli 4 görsel ile başlayın:

1. `sample-biometric.jpg` - Hero bölümü için örnek fotoğraf
2. `steps/step-1-upload.png` - Nasıl çalışır adım 1
3. `steps/step-2-process.png` - Nasıl çalışır adım 2  
4. `steps/step-3-download.png` - Nasıl çalışır adım 3

Bu 4 görsel ile site çok daha profesyonel görünecek!
