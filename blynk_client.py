"""التواصل مع Blynk عبر HTTP API (بدون ESP32).

- الإرسال في Thread مستقل مع طابور وإعادة محاولة: لا يتجمد الفيديو إذا انقطع الإنترنت.
- الرسائل المهمة (طلبات المريض والإشعارات) تُحفظ على القرص إن فشل إرسالها، وتُعاد عند عودة الاتصال.
- نبض حياة دوري (Heartbeat) ليعرف مقدّم الرعاية أن النظام يعمل.
- قراءة زر "تم الاستلام" من تطبيق مقدّم الرعاية (حلقة تواصل مغلقة).
"""
import json
import os
import queue
import threading
import time

import requests

import config
from app_logging import log


class Blynk:
    def __init__(self):
        self.enabled = config.BLYNK_ENABLED and bool(config.BLYNK_TOKEN)
        self.status = "off" if not self.enabled else "idle"   # off | idle | ok | error
        self._queue = queue.Queue(maxsize=200)
        self._last_sent = {}
        self._last_alert = -1e9
        self._last_heartbeat = -1e9
        self._awaiting_ack_until = 0.0
        self._ack = threading.Event()
        self._file_lock = threading.Lock()
        self._stop = threading.Event()
        if self.enabled:
            self._load_pending()
            threading.Thread(target=self._worker, daemon=True).start()
            threading.Thread(target=self._ack_poller, daemon=True).start()

    # ---------- HTTP ----------
    def _request(self, path, params):
        """طلب واحد مع إعادة المحاولة. تُرجع نص الرد عند النجاح، أو None."""
        bare = [k for k, v in params.items() if v is None]          # صيغة القراءة في Blynk: get?token=..&V4
        params = {"token": config.BLYNK_TOKEN, **{k: v for k, v in params.items() if v is not None}}
        url = f"{config.BLYNK_SERVER}/external/api/{path}" + ("?" + "&".join(bare) if bare else "")
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=5)
                if r.status_code == 200:
                    self.status = "ok"
                    return r.text
                log.warning(f"خطأ {r.status_code}: {r.text[:120]}")
                self.status = "error"
                if r.status_code in (400, 401, 403):     # خطأ دائم (توكن/pin خاطئ): لا فائدة من الإعادة
                    return None
            except requests.RequestException as e:
                log.warning(f"فشل الاتصال: {type(e).__name__}")
                self.status = "error"
            if self._stop.wait(config.BLYNK_RETRY_BASE_SECONDS * 2 ** attempt):
                return None
        return None

    # ---------- العامل الخلفي ----------
    def _worker(self):
        while not self._stop.is_set():
            try:
                path, params, persist = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            ok = self._request(path, params) is not None
            if not ok and persist:
                self._save_pending(path, params)
            elif ok:
                self._flush_pending()

    def _ack_poller(self):
        # نقرأ الزر فقط أثناء انتظار تأكيد رسالة (توفير رسائل الخطة المجانية في Blynk)
        while not self._stop.wait(config.BLYNK_ACK_POLL_SECONDS):
            if time.time() > self._awaiting_ack_until:
                continue
            value = self._request("get", {config.BLYNK_PIN_ACK: None})
            if value is not None and value.strip().strip('[]"') == "1":
                self._awaiting_ack_until = 0
                self._ack.set()
                self._send("update", {config.BLYNK_PIN_ACK: 0})   # إعادة الزر لوضعه الطبيعي

    # ---------- الطابور المحفوظ على القرص ----------
    def _save_pending(self, path, params):
        with self._file_lock:
            os.makedirs(os.path.dirname(config.BLYNK_PENDING_FILE) or ".", exist_ok=True)
            with open(config.BLYNK_PENDING_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"path": path, "params": params, "t": time.time()}, ensure_ascii=False) + "\n")
        log.warning("حُفظت الرسالة لإعادة إرسالها عند عودة الاتصال")

    def _load_pending(self):
        """عند البدء: أعد إرسال ما لم يُرسل في الجلسة السابقة."""
        self._flush_pending()

    def _flush_pending(self):
        with self._file_lock:
            if not os.path.exists(config.BLYNK_PENDING_FILE):
                return
            with open(config.BLYNK_PENDING_FILE, encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]
            os.remove(config.BLYNK_PENDING_FILE)
        for it in items:
            params = dict(it["params"])
            if it["path"] == "logEvent":
                age = int((time.time() - it["t"]) / 60)
                if age >= 1:
                    params["description"] = f"{params.get('description', '')} (متأخرة {age} د)"
            self._send(it["path"], params, persist=True)

    def _send(self, path, params, persist=False):
        if not self.enabled:
            return
        try:
            self._queue.put_nowait((path, params, persist))
        except queue.Full:
            if persist:
                self._save_pending(path, params)

    # ---------- واجهة الاستخدام ----------
    def write(self, pin, value, min_interval=1.0, persist=False):
        """تحديث Datastream (مع منع الإرسال المتكرر بسرعة)."""
        now = time.time()
        if now - self._last_sent.get(pin, 0) < min_interval:
            return
        self._last_sent[pin] = now
        self._send("update", {pin: value}, persist)

    def alert(self, description, force=False):
        """إشعار على تطبيق Blynk. force=True للطوارئ (يتجاوز الفاصل الأدنى)."""
        now = time.time()
        if not force and now - self._last_alert < config.BLYNK_ALERT_MIN_INTERVAL:
            return
        self._last_alert = now
        self._awaiting_ack_until = now + config.BLYNK_ACK_WAIT_SECONDS
        self._send("logEvent", {"code": config.BLYNK_EVENT_CODE, "description": description}, persist=True)

    def heartbeat(self):
        """يُستدعى في كل دورة؛ يرسل فعلياً كل BLYNK_HEARTBEAT_SECONDS."""
        now = time.time()
        if now - self._last_heartbeat >= config.BLYNK_HEARTBEAT_SECONDS:
            self._last_heartbeat = now
            self._send("update", {config.BLYNK_PIN_HEARTBEAT: time.strftime("%H:%M:%S")})

    def consume_ack(self):
        """True مرة واحدة إذا ضغط مقدّم الرعاية "تم الاستلام" منذ آخر استدعاء."""
        if self._ack.is_set():
            self._ack.clear()
            return True
        return False

    def close(self):
        self._stop.set()
