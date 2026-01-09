#!/usr/bin/env python3
"""Test the latest uploaded photo"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['USE_V2_ANALYZER'] = 'true'

from utils.analyze_v2 import analyze_image_v2

# Find latest upload
upload_files = glob.glob('uploads/*.{jpg,jpeg,png,webp}')
if not upload_files:
    print("No uploads found")
    sys.exit(1)

latest = max(upload_files, key=os.path.getmtime)
print(f"Testing latest upload: {latest}")

result = analyze_image_v2("test", latest)

issues = result.get("issues", [])
metrics = result.get("metrics", {})

print(f"\n📊 Metrics:")
print(f"  sunglasses_score: {metrics.get('sunglasses_score', 'N/A')}")
print(f"  iris_visibility: {metrics.get('iris_visibility', 'N/A')}")
print(f"  hair_occlusion_score: {metrics.get('hair_occlusion_score', 'N/A')}")

print(f"\n🚨 Issues:")
for issue in issues:
    code = issue.get("code", "UNKNOWN")
    message = issue.get("message", "")
    print(f"  [{issue.get('severity', 'unknown').upper()}] {code}: {message}")

has_sunglasses = any(i.get("code") == "SUNGLASSES" for i in issues)
has_hair = any(i.get("code") == "HAIR_OVER_EYES" for i in issues)

print(f"\n✅ Result:")
print(f"  Sunglasses: {has_sunglasses}")
print(f"  Hair: {has_hair}")

if has_sunglasses and not has_hair:
    print("\n✅✅✅ CORRECT!")
elif has_sunglasses and has_hair:
    print("\n❌❌❌ ERROR: Both detected!")
elif not has_sunglasses and has_hair:
    print("\n❌❌❌ ERROR: Hair but no sunglasses!")

