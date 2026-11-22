"""توليد الملفات الصوتية لكل العبارات (مرة واحدة، يحتاج إنترنت) بصوت عربي عصبي عبر edge-tts.

    python generate_audio.py                 # يولّد الناقص فقط
    python generate_audio.py --voice ar-IQ-RanaNeural --force

ملاحظة خصوصية: تُرسل نصوص العبارات العامة فقط ("أحتاج ماء" …) إلى خدمة Microsoft لتحويلها لصوت.
لاستخدام صوت حقيقي (أحد أفراد العائلة مثلاً): سجّل العبارة mp3 وضعها بنفس اسم الملف المذكور في
assets/audio/index.json.
"""
import argparse
import asyncio
import json
import os

import edge_tts

import config
from alerts import audio_filename
from patient_profile import load_profile
import patient_profile
from scan_menu import DEFAULT_MENU


def phrases():
    out = list(config.BLINK_MESSAGES.values()) + [config.LONG_BLINK_MESSAGE, config.ACK_PHRASE]
    out += [p for c in (patient_profile.CUSTOM_MENU or DEFAULT_MENU) for p in c["items"]]
    return list(dict.fromkeys(out))          # إزالة التكرار مع الحفاظ على الترتيب


async def generate(voice, force):
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    index_path = os.path.join(config.AUDIO_DIR, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    made = 0
    for p in phrases():
        name = index.get(p) or audio_filename(p)
        path = os.path.join(config.AUDIO_DIR, name)
        index[p] = name
        if os.path.exists(path) and not force:
            continue
        await edge_tts.Communicate(p, voice, rate="-10%").save(path)
        made += 1
        print(f"  ✓ {p}  ->  {name}")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"تم: {made} ملف جديد، {len(index)} عبارة في {index_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=config.AUDIO_VOICE)
    ap.add_argument("--profile", default="patients/default.json")
    ap.add_argument("--force", action="store_true", help="إعادة توليد الموجود")
    a = ap.parse_args()
    if a.profile and os.path.exists(a.profile):
        load_profile(a.profile)
    asyncio.run(generate(a.voice, a.force))
