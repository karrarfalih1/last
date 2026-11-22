"""تتبع الوجه بـ MediaPipe FaceLandmarker (الواجهة الأحدث):
  - 478 نقطة للوجه (للرمش EAR ولقصّ الوجه للمشاعر)
  - Blendshapes: احتمال إغلاق كل عين (eyeBlinkLeft/Right) — أقل تأثراً بزاوية الرأس من EAR
  - مصفوفة التحويل: زاوية الرأس (yaw/pitch/roll) لتجاهل الرمش عند الالتفات الكبير
"""
import math
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config
from blink_detector import face_ear


@dataclass
class FaceObs:
    points: np.ndarray      # (478, 2) بالبكسل
    ear: float              # Eye Aspect Ratio (هندسي)
    blink_score: float      # 0 مفتوحة … 1 مغلقة (من الشبكة)
    yaw: float              # درجات
    pitch: float
    roll: float

    def openness(self, signal=None):
        """قيمة "انفتاح العين" التي يعاير عليها EyeStateTracker (أقل = أكثر إغلاقاً)."""
        signal = signal or config.BLINK_SIGNAL
        if signal == "ear":
            return self.ear
        if signal == "blendshape":
            return 1.0 - self.blink_score
        # fused: متوسط الإشارتين بعد تطبيع EAR إلى [0..1].
        # السقف مهم: ميل الرأس يضغط عرض العين فيُضخّم EAR (قد يتجاوز 0.8)، ولو تُرك بلا سقف
        # لتضخّم خط الأساس أثناء المعايرة، ثم بدت العين "مغلقة" عند أي حركة للرأس.
        ear_n = min(self.ear / 0.30, 1.0)
        return 0.5 * ear_n + 0.5 * (1.0 - self.blink_score)


def rotation_to_euler(m):
    """yaw, pitch, roll بالدرجات من مصفوفة تحويل 4×4."""
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class FaceTracker:
    def __init__(self, model_path=None):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path or config.FACE_LANDMARKER_PATH),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, frame_bgr, timestamp_ms=None):
        """تُرجع FaceObs أو None إن لم يوجد وجه."""
        ts = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        ts = max(ts, self._last_ts + 1)              # VIDEO mode يتطلب توقيتاً متزايداً
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        r = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not r.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in r.face_landmarks[0]])
        bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
        yaw, pitch, roll = rotation_to_euler(r.facial_transformation_matrixes[0])
        return FaceObs(pts, face_ear(pts), (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2, yaw, pitch, roll)

    def close(self):
        self._lm.close()


class PoseGate:
    """يتعلم وضع الرأس الطبيعي أثناء المعايرة، ثم يرفض الإطارات التي يلتفت فيها الرأس كثيراً
    (الالتفات/النظر للأسفل يخفض EAR فيبدو كإغلاق للعين = إنذار كاذب)."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self._samples = []
        self.yaw0 = self.pitch0 = None

    def collect(self, obs):
        self._samples.append((obs.yaw, obs.pitch))

    def finish(self):
        if self._samples:
            self.yaw0, self.pitch0 = np.median(self._samples, axis=0)
        self._samples = []

    def ok(self, obs):
        if not self.enabled or self.yaw0 is None:
            return True
        return (abs(obs.yaw - self.yaw0) <= config.MAX_YAW_DEVIATION
                and abs(obs.pitch - self.pitch0) <= config.MAX_PITCH_DEVIATION)
