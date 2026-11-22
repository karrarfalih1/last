"""تمييز المشاعر بنموذج CNN، مع:
  1) تثبيت النتيجة بمتوسط آخر عدة توقعات (يمنع القفز بين المشاعر).
  2) معايرة شخصية: نتعلّم شكل الوجه الهادئ للشخص في أول ثوانٍ، ثم نقيس الانحراف عنه.
     نماذج FER-2013 كثيراً ما تقرأ الوجه الهادئ "حزين/غاضب"، فبدون هذا الإجراء تصل إنذارات كاذبة."""
import os
from collections import deque

import cv2
import numpy as np

import config
from app_logging import log

NEUTRAL = config.EMOTION_LABELS.index("neutral")
EPS = 0.03


def baseline_adjust(probs, baseline, neutral_bias=None):
    """تُرجع توزيعاً جديداً: كل فئة تُقاس بنسبة ظهورها مقارنةً بمستواها وقت الراحة (log-ratio)،
    مع تفضيل بسيط لـ neutral، فيكون الوجه الهادئ = طبيعي، وأي ارتفاع واضح يتفوق."""
    bias = config.EMOTION_NEUTRAL_BIAS if neutral_bias is None else neutral_bias
    z = np.log(probs + EPS) - np.log(baseline + EPS)
    z[NEUTRAL] += bias
    e = np.exp(z - z.max())
    return e / e.sum()


class EmotionDetector:
    def __init__(self):
        self.model = None
        self.history = deque(maxlen=config.EMOTION_SMOOTHING)
        self.baseline = None
        self._baseline_samples = []
        path = config.EMOTION_MODEL_PATH
        if os.path.exists(path):
            if path.endswith(".tflite"):          # أسرع على المعالج (انظر export_tflite.py)
                import tensorflow as tf
                self._tfl = tf.lite.Interpreter(model_path=path, num_threads=2)
                self._tfl.allocate_tensors()
                self._tfl_in = self._tfl.get_input_details()[0]["index"]
                self._tfl_out = self._tfl.get_output_details()[0]["index"]
                self.model = self._tfl
            else:
                from tensorflow import keras
                self._tfl = None
                self.model = keras.models.load_model(path)
            self._infer(np.zeros((1, 48, 48, 1), "float32"))   # تسخين أول استدعاء
        else:
            log.warning(f"لم يُعثر على {config.EMOTION_MODEL_PATH} — شغّل train_emotion.py. "
                  "سيعمل كشف الرمش فقط.")

    @property
    def available(self):
        return self.model is not None

    def reset(self):
        """إعادة المعايرة (مثلاً عند الضغط على R)."""
        self.history.clear()
        self.baseline = None
        self._baseline_samples = []

    @staticmethod
    def crop_face(frame, points):
        h, w = frame.shape[:2]
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        mx, my = (x2 - x1) * config.EMOTION_FACE_MARGIN, (y2 - y1) * config.EMOTION_FACE_MARGIN
        x1, y1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
        x2, y2 = int(min(x2 + mx, w)), int(min(y2 + my, h))
        return frame[y1:y2, x1:x2]

    def _raw_probs(self, frame, points):
        face = self.crop_face(frame, points)
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # ملاحظة: النموذج يحتوي طبقة Rescaling(1/255) داخلياً، فنمرّر القيم الخام 0–255 (لا نقسم هنا!)
        img = cv2.resize(gray, (48, 48)).astype("float32")
        return self._infer(img.reshape(1, 48, 48, 1))

    def _infer(self, batch):
        if self._tfl is not None:
            self._tfl.set_tensor(self._tfl_in, batch)
            self._tfl.invoke()
            return self._tfl.get_tensor(self._tfl_out)[0]
        return self.model(batch, training=False).numpy()[0]   # أسرع من predict() لصورة واحدة

    # ---------- المعايرة ----------
    def collect_baseline(self, frame, points):
        """يُستدعى أثناء المعايرة (الشخص بوجه هادئ)."""
        if not self.available:
            return
        probs = self._raw_probs(frame, points)
        if probs is not None:
            self._baseline_samples.append(probs)

    def finish_baseline(self):
        if self._baseline_samples:
            self.baseline = np.mean(self._baseline_samples, axis=0)
        self._baseline_samples = []
        self.history.clear()

    # ---------- التوقع ----------
    def predict(self, frame, points):
        """تُرجع (label, confidence, probs) بعد التثبيت والمعايرة الشخصية، أو None."""
        if not self.available:
            return None
        probs = self._raw_probs(frame, points)
        if probs is None:
            return None
        self.history.append(probs)
        mean = np.mean(self.history, axis=0)
        if self.baseline is not None:
            mean = baseline_adjust(mean, self.baseline)
        i = int(np.argmax(mean))
        return config.EMOTION_LABELS[i], float(mean[i]), mean
