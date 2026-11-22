"""الأصوات المحلية: صفارات التنبيه + نطق الرسائل بملفات صوتية مسبقة التوليد.

كل الأصوات تمر بطابور واحد فلا تتداخل (الصفارة ثم العبارة). التشغيل بدون إنترنت:
ملفات mp3 في assets/audio تُشغَّل بمشغّل ويندوز المدمج (MCI)، دون ffmpeg.
لتوليد الملفات:  python generate_audio.py     (يمكن استبدال أي ملف بتسجيل صوت حقيقي بنفس الاسم)
"""
import hashlib
import json
import os
import queue
import threading
import time

import config
from app_logging import log

try:
    import winsound
except ImportError:      # ليس ويندوز
    winsound = None

BEEPS = {
    "emergency": [(1500, 250)] * 3,
    "ack": [(660, 150), (880, 150), (1100, 220)],     # نغمة صاعدة مطمئنة
    "select": [(1000, 80)],                           # نقرة قصيرة عند الاختيار في القائمة
    "menu": [(700, 120), (900, 120)],
}


def audio_filename(phrase):
    return hashlib.md5(phrase.encode("utf-8")).hexdigest()[:12] + ".mp3"


def _play_mp3(path):
    """تشغيل mp3 وانتظار انتهائه عبر Windows MCI (winmm)."""
    import ctypes
    mci = ctypes.windll.winmm.mciSendStringW
    alias = f"snd{threading.get_ident()}"
    if mci(f'open "{os.path.abspath(path)}" type mpegvideo alias {alias}', None, 0, None) == 0:
        mci(f"play {alias} wait", None, 0, None)
        mci(f"close {alias}", None, 0, None)


class LocalAlert:
    def __init__(self):
        self.muted = not config.LOCAL_SOUND
        self._q = queue.Queue(maxsize=20)
        self._index = {}
        index_path = os.path.join(config.AUDIO_DIR, "index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                self._index = json.load(f)
        threading.Thread(target=self._worker, daemon=True).start()

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    def has_voice(self, phrase):
        return self._path(phrase) is not None

    def _path(self, phrase):
        name = self._index.get(phrase) or audio_filename(phrase)
        path = os.path.join(config.AUDIO_DIR, name)
        return path if os.path.exists(path) else None

    def _put(self, item):
        if self.muted:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass

    def play(self, kind):
        self._put(("beep", kind))

    def say(self, phrase):
        if config.SPEAK_MESSAGES:
            path = self._path(phrase)
            if path:
                self._put(("file", path))

    def _worker(self):
        while True:
            kind, arg = self._q.get()
            try:
                if kind == "file" and os.name == "nt":
                    _play_mp3(arg)
                elif kind == "beep":
                    for freq, ms in BEEPS.get(arg, [(900, 200)]):
                        if winsound:
                            winsound.Beep(freq, ms)
                        else:
                            print("\a", end="", flush=True)
                        time.sleep(0.08)
            except Exception as e:          # الصوت ميزة مساعدة: لا يجب أن يوقف البرنامج
                log.warning(f"خطأ صوت: {e}")
