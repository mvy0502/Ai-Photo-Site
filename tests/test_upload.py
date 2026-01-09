#!/usr/bin/env python3
"""
Test script for photo upload functionality
"""
import requests
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_valid_image():
    """Test uploading a valid image file"""
    print("Test 1: Geçerli resim dosyası yükleme")
    # Bu test için gerçek bir resim dosyası gerekir
    # Örnek: test_image.jpg dosyası oluşturmanız gerekir
    print("⚠️  Bu test için bir resim dosyası gerekir")
    print()

def test_invalid_extension():
    """Test uploading file with invalid extension"""
    print("Test 2: Geçersiz uzantılı dosya")
    test_file = "test.txt"
    # Boş bir test dosyası oluştur
    with open(test_file, "w") as f:
        f.write("test content")
    
    try:
        with open(test_file, "rb") as f:
            files = {"photo": (test_file, f, "text/plain")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            print(f"Status: {response.status_code}")
            if "Sadece .jpg, .jpeg, .png ve .webp" in response.text:
                print("✅ Geçersiz uzantı kontrolü çalışıyor")
            else:
                print("❌ Geçersiz uzantı kontrolü çalışmıyor")
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    print()

def test_empty_file():
    """Test uploading empty file"""
    print("Test 3: Boş dosya yükleme")
    empty_file = "empty.txt"
    # Boş dosya oluştur
    Path(empty_file).touch()
    
    try:
        with open(empty_file, "rb") as f:
            files = {"photo": (empty_file, f, "text/plain")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            print(f"Status: {response.status_code}")
            if "Boş dosya yüklenemez" in response.text:
                print("✅ Boş dosya kontrolü çalışıyor")
            else:
                print("❌ Boş dosya kontrolü çalışmıyor")
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        if os.path.exists(empty_file):
            os.remove(empty_file)
    print()

def check_uploads_folder():
    """Check uploads folder contents"""
    print("Test 4: Uploads klasörü kontrolü")
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        files = list(uploads_dir.glob("*"))
        print(f"✅ Uploads klasöründe {len(files)} dosya var")
        for f in files[:5]:  # İlk 5 dosyayı göster
            print(f"   - {f.name}")
    else:
        print("❌ Uploads klasörü bulunamadı")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("AI Photo Site - Test Senaryoları")
    print("=" * 50)
    print()
    
    # Uygulamanın çalışıp çalışmadığını kontrol et
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✅ Uygulama çalışıyor")
        else:
            print(f"⚠️  Uygulama yanıt veriyor ama status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Uygulama çalışmıyor: {e}")
        print("   Lütfen önce uygulamayı başlatın: uvicorn app:app --reload")
        exit(1)
    
    print()
    test_invalid_extension()
    test_empty_file()
    check_uploads_folder()
    test_valid_image()
    
    print("=" * 50)
    print("Testler tamamlandı!")
    print("=" * 50)

