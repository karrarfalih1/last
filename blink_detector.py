"""كشف الرمش: حساب EAR، معايرة تلقائية بحدّين (hysteresis)، وتحويل نمط الرمشات إلى رسالة.
المنطق هنا لا يعتمد على الكاميرا ولا على الوقت الحقيقي (الوقت يُمرَّر كوسيط) فيمكن اختباره."""
from dataclasses import dataclass

import numpy as np

import config

# نقاط العين في MediaPipe Face Mesh (6 نقاط لكل عين)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(pts):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) — تقل القيمة كلما أُغلقت العين."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2.0 * horizontal + 1e-6))


def face_ear(points):
    """متوسط EAR للعينين من مصفوفة نقاط الوجه (468+ نقطة بالبكسل)."""
    return (eye_aspect_ratio(points[LEFT_EYE]) + eye_aspect_ratio(points[RIGHT_EYE])) / 2


class EyeStateTracker:
    """يعاير انفتاح العين الطبيعي للشخص، ثم يحدد مغلقة/مفتوحة بحدّين لمنع الاهتزاز.
    يعمل على أي إشارة "انفتاح" (EAR أو blendshape أو المدمجة) لأن الحدود نسبية من خط الأساس.
    بعد المعايرة يتكيّف خط الأساس ببطء مع الإطارات المفتوحة (تغيّر الإضاءة، التعب، تحرّك الكاميرا)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calibrating = True
        self._values = []
        self._start = None
        self.closed = False
        self.close_threshold = config.EAR_THRESHOLD
        self.open_threshold = config.EAR_THRESHOLD * 1.1
        self.baseline = None

    @property
    def calibration_remaining(self):
        return getattr(self, "_remaining", config.CALIBRATION_SECONDS)

    def update(self, ear, now):
        """تُرجع True إذا كانت العين مغلقة، وNone أثناء المعايرة."""
        if self.calibrating:
            if self._start is None:
                self._start = now
            self._values.append(ear)
            self._remaining = max(0.0, config.CALIBRATION_SECONDS - (now - self._start))
            if self._remaining <= 0 and len(self._values) >= 10:
                self.baseline = float(np.median(self._values))   # الوسيط يتجاهل الرمشات العرضية
                self._set_thresholds()
                self.calibrating = False
            return None

        if self.closed:
            if ear > self.open_threshold:
                self.closed = False
        elif ear < self.close_threshold:
            self.closed = True

        # تكيّف بطيء: فقط من إطارات مفتوحة بوضوح، فلا يتعلم الإغلاق المتعمّد
        rate = config.ADAPTIVE_BASELINE_RATE
        if rate and not self.closed and ear > self.open_threshold:
            self.baseline = (1 - rate) * self.baseline + rate * ear
            self._set_thresholds()
        return self.closed

    def _set_thresholds(self):
        self.close_threshold = self.baseline * config.CALIBRATION_CLOSE_RATIO
        self.open_threshold = max(self.baseline * config.CALIBRATION_OPEN_RATIO, self.close_threshold * 1.03)


@dataclass
class Message:
    text: str
    kind: str      # "request" طلب | "emergency" طوارئ | "menu" فتح القائمة | "select" اختيار في القائمة
    blinks: int    # عدد الرمشات التي كوّنت النمط


class BlinkInterpreter:
    """آلة حالات: إغلاق/فتح العين -> رمشات متعمّدة -> رسالة.
    mode="pattern": تُجمع الرمشات في نمط (2 = ماء …).
    mode="select" : كل رمشة متعمّدة = اختيار فوري (لقائمة المسح). الطوارئ تعمل في الوضعين."""

    def __init__(self):
        self.mode = "pattern"
        self.blink_count = 0
        self.ignored_blinks = 0        # رمشات لاإرادية قصيرة تم تجاهلها
        self.closed_since = None
        self.last_blink_time = 0.0
        self.long_fired = False
        self.last_emergency = -1e9

    # ---- حالة للعرض في الواجهة ----
    def closed_for(self, now):
        return 0.0 if self.closed_since is None else now - self.closed_since

    def pattern_time_left(self, now):
        if not self.blink_count:
            return 0.0
        return max(0.0, config.PATTERN_TIMEOUT - (now - self.last_blink_time))

    def reset(self):
        self.blink_count = 0
        self.closed_since = None
        self.long_fired = False

    def update(self, closed, now):
        """closed: True/False، أو None إذا اختفى الوجه. تُرجع Message عند اكتمال نمط."""
        if closed is None:                       # اختفى الوجه: لا نحسب الإغلاق كرمشة
            self.closed_since = None
            self.long_fired = False
            return self._finish_pattern(now)

        if closed:
            if self.closed_since is None:
                self.closed_since = now
            if not self.long_fired and now - self.closed_since >= config.LONG_BLINK_SECONDS:
                self.long_fired = True
                self.blink_count = 0
                if now - self.last_emergency >= config.EMERGENCY_COOLDOWN:
                    self.last_emergency = now
                    return Message(config.LONG_BLINK_MESSAGE, "emergency", 0)
            return None

        # العين مفتوحة
        if self.closed_since is not None:
            duration = now - self.closed_since
            deliberate = not self.long_fired and duration >= config.DELIBERATE_BLINK_MIN_SECONDS
            if not self.long_fired and not deliberate:
                self.ignored_blinks += 1
            self.closed_since = None
            self.long_fired = False
            if deliberate:
                if self.mode == "select":
                    self.blink_count = 0
                    return Message("", "select", 1)
                self.blink_count += 1
                self.last_blink_time = now
        return self._finish_pattern(now)

    def set_mode(self, mode):
        self.mode = mode
        self.blink_count = 0

    def _finish_pattern(self, now):
        if self.mode == "pattern" and self.blink_count and now - self.last_blink_time > config.PATTERN_TIMEOUT:
            count, self.blink_count = self.blink_count, 0
            if count == config.MENU_PATTERN:
                return Message("", "menu", count)
            text = config.BLINK_MESSAGES.get(count)
            if text:
                return Message(text, "request", count)
        return None
