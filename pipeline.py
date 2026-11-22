"""خط معالجة الرمش الكامل لإطار واحد: ملاحظة الوجه -> معايرة -> فحص زاوية الرأس -> حالة العين -> رسالة.
يُستخدم في main.py وفي evaluate_blink.py (فالتقييم يقيس نفس الكود الذي يعمل فعلاً)."""
import config
from blink_detector import BlinkInterpreter, EyeStateTracker
from face_tracker import PoseGate


class BlinkPipeline:
    def __init__(self, signal=None, pose_gate=True):
        self.signal = signal or config.BLINK_SIGNAL
        self.eye = EyeStateTracker()
        self.pose = PoseGate(enabled=pose_gate)
        self.blinks = BlinkInterpreter()
        self.closed = False
        self.pose_ok = True
        self.value = 0.0
        self.calib_warning = ""        # يُعرض للمستخدم إن فشلت المعايرة
        self.calib_note = ""           # ملاحظة غير مانعة (مثلاً: الوجه مائل كثيراً)

    def reset(self):
        warn = self.calib_warning
        self.__init__(self.signal, self.pose.enabled)
        self.calib_warning = warn

    @property
    def calibrating(self):
        return self.eye.calibrating

    def step(self, obs, now):
        """obs: FaceObs أو None. تُرجع Message عند اكتمال نمط، وإلا None."""
        if obs is None:
            return self.blinks.update(None, now)

        self.value = obs.openness(self.signal)
        was_calibrating = self.eye.calibrating
        closed = self.eye.update(self.value, now)
        if self.eye.calibrating:
            self.pose.collect(obs)
            return None
        if was_calibrating:                 # انتهت المعايرة في هذا الإطار
            self.pose.finish()
            # فحص الجودة: يجب أن تُلتقط العينان مفتوحتين (الإشارة المعتمدة على الشبكة محصورة في 0..1)
            if self.signal != "ear" and self.eye.baseline < config.MIN_CALIBRATION_BASELINE:
                self.calib_warning = "لم تُلتقط عيناك مفتوحتين بوضوح — افتحهما جيداً وواجه الكاميرا"
                self.eye.reset()
                self.pose.reset()
                return None
            self.calib_warning = ""
            self.calib_note = ("الوجه مائل كثيراً عن الكاميرا؛ قد تقل الدقة"
                               if abs(self.pose.yaw0) > config.WARN_CALIBRATION_YAW else "")
            return None

        self.pose_ok = self.pose.ok(obs)
        if not self.pose_ok:                # الرأس ملتفت: القراءة غير موثوقة، لا نحسب إغلاقاً
            self.closed = False
            return self.blinks.update(None, now)
        self.closed = closed
        return self.blinks.update(closed, now)
