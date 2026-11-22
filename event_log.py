"""سجل أحداث الجلسة في ملف CSV (يفيد في التقرير وتحليل الأداء)."""
import csv
import os
import time


class EventLog:
    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, time.strftime("session_%Y%m%d_%H%M%S.csv"))
        self._file = open(self.path, "w", newline="", encoding="utf-8-sig")   # utf-8-sig ليفتح بالعربية في Excel
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time", "type", "detail"])
        self._file.flush()

    def log(self, kind, detail=""):
        self._writer.writerow([time.strftime("%H:%M:%S"), kind, detail])
        self._file.flush()

    def close(self):
        self._file.close()
