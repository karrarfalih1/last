"""قراءة الكاميرا في Thread منفصل: دائماً أحدث إطار (بدون تأخير متراكم)، وإعادة اتصال تلقائية."""
import os
import threading
import time

import cv2


class Camera:
    def __init__(self, source):
        self.source = source
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = True
        self.connected = False
        self.cap = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        cap = None
        if isinstance(self.source, int) and os.name == "nt":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)   # أسرع في الفتح على ويندوز
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return cap

    def _loop(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read() if self.cap.isOpened() else (False, None)
            if ok:
                failures = 0
                self.connected = True
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
                continue
            failures += 1
            self.connected = False
            if failures > 30:                      # انقطع البث: أعد الاتصال
                self.cap.release()
                time.sleep(1.0)
                self.cap = self._open()
                failures = 0
            else:
                time.sleep(0.03)

    def read(self, last_seq=-1):
        """تُرجع (إطار, رقمه) للإطار الأحدث، أو (None, last_seq) إن لم يصل جديد."""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None, last_seq
            return self._frame.copy(), self._seq

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()
