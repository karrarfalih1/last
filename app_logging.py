"""تسجيل موحّد: الشاشة + ملف logs/app.log (يدور تلقائياً)، وحذف السجلات القديمة (خصوصية)."""
import glob
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import config

log = logging.getLogger("assist")


def setup(level=logging.INFO):
    if log.handlers:
        return log
    os.makedirs(config.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(module)s] %(message)s", "%H:%M:%S")
    try:
        sys.stdout.reconfigure(encoding="utf-8")       # العربية في طرفية ويندوز
    except (AttributeError, ValueError):
        pass
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file = RotatingFileHandler(os.path.join(config.LOG_DIR, "app.log"), maxBytes=2_000_000,
                               backupCount=3, encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(module)s] %(message)s"))
    log.addHandler(console)
    log.addHandler(file)
    log.setLevel(level)
    return log


def cleanup_old_logs(days=None):
    """يحذف سجلات الجلسات ولقطات الشاشة الأقدم من LOG_RETENTION_DAYS. تُرجع عدد المحذوف."""
    days = config.LOG_RETENTION_DAYS if days is None else days
    if not days:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for pattern in ("session_*.csv", "shot_*.png"):
        for path in glob.glob(os.path.join(config.LOG_DIR, pattern)):
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
    return removed
