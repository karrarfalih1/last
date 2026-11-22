"""قائمة المسح (Scanning Menu) — أسلوب التواصل البديل المعتمد (AAC) لمن يملك "مفتاحاً" واحداً فقط.

المؤشر ينتقل تلقائياً بين الخيارات، والمريض يرمش رمشة متعمّدة واحدة عندما يُضاء الخيار المطلوب.
مستويان: فئات -> عبارات. تُفتح بنمط config.MENU_PATTERN (5 رمشات افتراضياً).
المنطق هنا لا يعتمد على الوقت الحقيقي (يُمرَّر now) فيمكن اختباره.
"""
import config

CLOSE = "إغلاق القائمة"
BACK = "رجوع"

DEFAULT_MENU = [
    {"title": "احتياجات", "items": ["أحتاج ماء", "أحتاج طعام", "أحتاج الحمام", "أريد تغيير وضعيتي",
                                   "أشعر بالبرد", "أشعر بالحر", "أريد أن أنام"]},
    {"title": "ألم", "items": ["ألم في الرأس", "ألم في الصدر", "ألم في البطن", "ألم في الظهر",
                              "ألم في اليد", "ألم في الساق"]},
    {"title": "أشخاص", "items": ["أريد أمي", "أريد أبي", "أريد الممرض", "أريد الطبيب"]},
    {"title": "ردود", "items": ["نعم", "لا", "شكراً", "أنا بخير", "لا أعرف"]},
]


class ScanMenu:
    def __init__(self, menu=None):
        self.menu = menu or DEFAULT_MENU
        self.active = False
        self.category = None        # None = المستوى الأعلى
        self.index = 0
        self._last_step = 0.0
        self._cycles = 0

    # ---------- الحالة ----------
    def items(self):
        if self.category is None:
            return [c["title"] for c in self.menu] + [CLOSE]
        return list(self.menu[self.category]["items"]) + [BACK]

    @property
    def title(self):
        return "القائمة" if self.category is None else self.menu[self.category]["title"]

    @property
    def current(self):
        return self.items()[self.index]

    def progress(self, now):
        """0..1: كم مضى من وقت الخيار الحالي (لشريط العدّ في الواجهة)."""
        return min(1.0, (now - self._last_step) / config.SCAN_INTERVAL)

    # ---------- التحكم ----------
    def open(self, now):
        self.active, self.category = True, None
        self._reset_scan(now)

    def close(self):
        self.active = False

    def _reset_scan(self, now):
        self.index, self._cycles, self._last_step = 0, 0, now

    def tick(self, now, paused=False):
        """يُستدعى كل إطار. paused=True أثناء إغلاق العين: لا ينتقل المؤشر،
        فيبقى مضاءً الخيار الذي بدأ المريض الرمش عنده."""
        if not self.active:
            return
        if paused:
            self._last_step = now - min(now - self._last_step, config.SCAN_INTERVAL * 0.99)
            return
        if now - self._last_step >= config.SCAN_INTERVAL:
            self._last_step = now
            self.index = (self.index + 1) % len(self.items())
            if self.index == 0:
                self._cycles += 1
                if self._cycles >= config.SCAN_MAX_CYCLES:
                    self.close()

    def select(self, now):
        """رمشة اختيار. تُرجع العبارة المطلوب إرسالها، أو None (دخول فئة/رجوع/إغلاق)."""
        if not self.active:
            return None
        choice = self.current
        if self.category is None:
            if choice == CLOSE:
                self.close()
            else:
                self.category = self.index
                self._reset_scan(now)
            return None
        if choice == BACK:
            self.category = None
            self._reset_scan(now)
            return None
        self.close()
        return choice

    def all_phrases(self):
        return [p for c in self.menu for p in c["items"]]
