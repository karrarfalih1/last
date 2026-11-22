"""اختبارات آلية بدون كاميرا:  python -m unittest discover -s tests -v   (من مجلد المشروع)"""
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from blink_detector import BlinkInterpreter, EyeStateTracker

FPS = 30


def simulate(interp, script, t0=100.0):
    """script: قائمة (حالة, مدة بالثواني) حيث الحالة True=مغلقة False=مفتوحة None=لا وجه. تُرجع الرسائل الناتجة."""
    t, out = t0, []
    for closed, dur in script:
        for _ in range(max(1, round(dur * FPS))):
            m = interp.update(closed, t)
            if m:
                out.append(m)
            t += 1 / FPS
    return out


class BlinkPatternTests(unittest.TestCase):
    def test_two_deliberate_blinks_means_water(self):
        msgs = simulate(BlinkInterpreter(), [(False, 1), (True, 0.35), (False, 0.3), (True, 0.35), (False, 2.5)])
        self.assertEqual([m.text for m in msgs], [config.BLINK_MESSAGES[2]])

    def test_three_blinks_means_food(self):
        script = [(False, 1)] + [(True, 0.35), (False, 0.3)] * 3 + [(False, 2.5)]
        msgs = simulate(BlinkInterpreter(), script)
        self.assertEqual([m.text for m in msgs], [config.BLINK_MESSAGES[3]])

    def test_involuntary_short_blinks_are_ignored(self):
        interp = BlinkInterpreter()
        msgs = simulate(interp, [(False, 1)] + [(True, 0.1), (False, 0.3)] * 4 + [(False, 2.5)])
        self.assertEqual(msgs, [])
        self.assertEqual(interp.ignored_blinks, 4)

    def test_single_deliberate_blink_sends_nothing(self):
        self.assertEqual(simulate(BlinkInterpreter(), [(False, 1), (True, 0.4), (False, 3)]), [])

    def test_long_closure_is_emergency_once(self):
        msgs = simulate(BlinkInterpreter(), [(False, 1), (True, 3.0), (False, 2)])
        self.assertEqual([(m.kind, m.text) for m in msgs], [("emergency", config.LONG_BLINK_MESSAGE)])

    def test_emergency_cooldown(self):
        interp = BlinkInterpreter()
        first = simulate(interp, [(True, 2.0), (False, 1.0), (True, 2.0), (False, 1.0)])
        self.assertEqual(len(first), 1)      # الثانية خلال فترة الهدوء
        later = simulate(interp, [(False, config.EMERGENCY_COOLDOWN), (True, 2.0)], t0=200.0)
        self.assertEqual(len(later), 1)

    def test_long_closure_does_not_also_count_as_blink(self):
        msgs = simulate(BlinkInterpreter(), [(True, 2.0), (False, 3.0)])
        self.assertEqual([m.kind for m in msgs], ["emergency"])

    def test_face_loss_mid_closure_is_not_a_blink(self):
        interp = BlinkInterpreter()
        msgs = simulate(interp, [(False, 1), (True, 0.4), (None, 3.0)])
        self.assertEqual(msgs, [])
        self.assertEqual(interp.blink_count, 0)

    def test_pattern_completes_even_if_face_lost_after_blinks(self):
        script = [(False, 1), (True, 0.35), (False, 0.3), (True, 0.35), (False, 0.2), (None, 3.0)]
        msgs = simulate(BlinkInterpreter(), script)
        self.assertEqual([m.text for m in msgs], [config.BLINK_MESSAGES[2]])


class EyeStateTests(unittest.TestCase):
    def calibrated(self, baseline=0.34):
        tr = EyeStateTracker()
        t = 0.0
        while tr.calibrating:
            self.assertIsNone(tr.update(baseline, t))
            t += 1 / FPS
        return tr

    def test_calibration_sets_threshold_from_baseline(self):
        tr = self.calibrated(0.34)
        self.assertAlmostEqual(tr.close_threshold, 0.34 * config.CALIBRATION_CLOSE_RATIO, places=3)
        self.assertGreater(tr.open_threshold, tr.close_threshold)

    def test_calibration_ignores_blinks_during_calibration(self):
        tr = EyeStateTracker()
        t = 0.0
        while tr.calibrating:
            tr.update(0.10 if int(t * FPS) % 20 == 0 else 0.34, t)   # رمشة كل 20 إطار
            t += 1 / FPS
        self.assertAlmostEqual(tr.baseline, 0.34, places=2)

    def test_hysteresis_prevents_flicker(self):
        tr = self.calibrated(0.34)
        mid = (tr.close_threshold + tr.open_threshold) / 2
        self.assertFalse(tr.update(0.34, 10))
        self.assertTrue(tr.update(tr.close_threshold - 0.02, 11))
        self.assertTrue(tr.update(mid, 12))            # بين الحدّين تبقى مغلقة
        self.assertFalse(tr.update(tr.open_threshold + 0.02, 13))
        self.assertFalse(tr.update(mid, 14))           # وتبقى مفتوحة


class PipelineTests(unittest.TestCase):
    """خط المعالجة الكامل بملاحظات وجه مصطنعة (بدون كاميرا)."""

    def obs(self, ear=0.30, blink=0.02, yaw=0.0, pitch=10.0):
        import numpy as np
        from face_tracker import FaceObs
        return FaceObs(np.zeros((478, 2)), ear, blink, yaw, pitch, 0.0)

    def run_script(self, pipe, script, t0=0.0):
        """script: (closed?, مدة, yaw)"""
        t, out = t0, []
        for closed, dur, yaw in script:
            for _ in range(max(1, round(dur * FPS))):
                o = self.obs(0.10, 0.85, yaw) if closed else self.obs(0.30, 0.02, yaw)
                m = pipe.step(o, t)
                if m:
                    out.append(m)
                t += 1 / FPS
        return out

    def calibrate(self, pipe):
        return self.run_script(pipe, [(False, config.CALIBRATION_SECONDS + 0.5, 0)])

    def test_fused_signal_detects_two_blinks(self):
        from pipeline import BlinkPipeline
        pipe = BlinkPipeline("fused")
        self.calibrate(pipe)
        self.assertFalse(pipe.calibrating)
        msgs = self.run_script(pipe, [(True, 0.35, 0), (False, 0.3, 0), (True, 0.35, 0), (False, 2.5, 0)], t0=10)
        self.assertEqual([m.text for m in msgs], [config.BLINK_MESSAGES[2]])

    def test_blinks_ignored_when_head_turned(self):
        from pipeline import BlinkPipeline
        pipe = BlinkPipeline("fused")
        self.calibrate(pipe)
        turned = config.MAX_YAW_DEVIATION + 15
        msgs = self.run_script(pipe, [(True, 0.35, turned), (False, 0.3, turned),
                                      (True, 0.35, turned), (False, 2.5, turned)], t0=10)
        self.assertEqual(msgs, [])
        self.assertFalse(pipe.pose_ok)

    def test_head_turn_with_closed_eyes_is_not_emergency(self):
        from pipeline import BlinkPipeline
        pipe = BlinkPipeline("ear")
        self.calibrate(pipe)
        msgs = self.run_script(pipe, [(True, 3.0, 50)], t0=10)
        self.assertEqual(msgs, [])

    def test_regression_turned_head_inflated_ear_gives_no_false_emergency(self):
        """الخلل الحقيقي (جلسة 15/11): معايرة بوجه مائل 50° ⇒ EAR مضخّم (≈1.4) ⇒ خط أساس 2.55،
        ثم عند عودة الرأس (EAR≈0.38) بدت العين مغلقة دائماً ⇒ طوارئ متكررة وبلا أنماط."""
        from pipeline import BlinkPipeline
        pipe = BlinkPipeline("fused")
        t, msgs = 0.0, []
        for _ in range(int((config.CALIBRATION_SECONDS + 0.5) * FPS)):
            pipe.step(self.obs(ear=1.4, blink=0.02, yaw=51), t); t += 1 / FPS
        self.assertFalse(pipe.calibrating)
        self.assertAlmostEqual(pipe.eye.baseline, 0.99, delta=0.02)          # ليس 2.55
        for _ in range(10 * FPS):                                           # الرأس يعود نحو الكاميرا
            m = pipe.step(self.obs(ear=0.38, blink=0.01, yaw=40), t); t += 1 / FPS
            if m:
                msgs.append(m)
        self.assertEqual(msgs, [])
        self.assertFalse(pipe.closed)

    def test_deliberate_blinks_still_detected_after_turned_head_calibration(self):
        from pipeline import BlinkPipeline
        pipe = BlinkPipeline("fused")
        t = 0.0
        for _ in range(int((config.CALIBRATION_SECONDS + 0.5) * FPS)):
            pipe.step(self.obs(ear=1.4, blink=0.02, yaw=51), t); t += 1 / FPS
        out = []
        for closed, dur in [(True, 0.35), (False, 0.3), (True, 0.35), (False, 2.5)]:
            for _ in range(round(dur * FPS)):
                o = self.obs(ear=0.45, blink=0.85, yaw=51) if closed else self.obs(ear=1.4, blink=0.02, yaw=51)
                m = pipe.step(o, t); t += 1 / FPS
                if m:
                    out.append(m.text)
        self.assertEqual(out, [config.BLINK_MESSAGES[2]])

    def test_calibration_rejected_when_eyes_not_seen_open(self):
        from pipeline import BlinkPipeline
        pipe = BlinkPipeline("fused")
        t = 0.0
        for _ in range(int((config.CALIBRATION_SECONDS + 0.5) * FPS)):      # عينان مغمضتان أثناء المعايرة
            pipe.step(self.obs(ear=0.10, blink=0.9), t); t += 1 / FPS
        self.assertTrue(pipe.calibrating, "يجب أن تعيد المعايرة من البداية")
        self.assertTrue(pipe.calib_warning)
        for _ in range(int((config.CALIBRATION_SECONDS + 0.5) * FPS)):      # ثم فتحهما
            pipe.step(self.obs(), t); t += 1 / FPS
        self.assertFalse(pipe.calibrating)
        self.assertEqual(pipe.calib_warning, "")

    def test_rotation_to_euler(self):
        import math
        import numpy as np
        from face_tracker import rotation_to_euler
        a = math.radians(30)        # دوران حول محور y (التفات)
        m = np.eye(4)
        m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        yaw, pitch, roll = rotation_to_euler(m)
        self.assertAlmostEqual(yaw, 30, places=3)
        self.assertAlmostEqual(pitch, 0, places=3)

    def test_adaptive_baseline_follows_slow_drift_but_not_closures(self):
        tr = EyeStateTracker()
        t = 0.0
        while tr.calibrating:
            tr.update(0.34, t); t += 1 / FPS
        for _ in range(60 * FPS):           # دقيقة: الإضاءة تغيرت فصار الانفتاح الطبيعي 0.31
            tr.update(0.31, t); t += 1 / FPS
        self.assertAlmostEqual(tr.baseline, 0.31, delta=0.01)
        before = tr.baseline
        for _ in range(10 * FPS):           # إغلاق متعمّد طويل لا يجب أن يُتعلَّم
            tr.update(0.08, t); t += 1 / FPS
        self.assertAlmostEqual(tr.baseline, before, places=6)


class ScanMenuTests(unittest.TestCase):
    def test_five_blinks_open_menu(self):
        msgs = simulate(BlinkInterpreter(), [(False, 1)] + [(True, 0.35), (False, 0.3)] * config.MENU_PATTERN
                        + [(False, 2.5)])
        self.assertEqual([m.kind for m in msgs], ["menu"])

    def test_select_mode_reports_each_deliberate_blink_immediately(self):
        interp = BlinkInterpreter()
        interp.set_mode("select")
        msgs = simulate(interp, [(False, 0.5), (True, 0.35), (False, 0.1), (True, 0.1), (False, 0.3)])
        self.assertEqual([m.kind for m in msgs], ["select"])       # القصيرة اللاإرادية لا تختار

    def test_emergency_still_works_in_select_mode(self):
        interp = BlinkInterpreter()
        interp.set_mode("select")
        msgs = simulate(interp, [(True, 2.0), (False, 1.0)])
        self.assertEqual([m.kind for m in msgs], ["emergency"])

    def test_navigate_category_and_choose_phrase(self):
        from scan_menu import ScanMenu
        m = ScanMenu()
        m.open(0.0)
        self.assertEqual(m.current, "احتياجات")
        self.assertIsNone(m.select(0.5))                 # دخول الفئة
        self.assertEqual(m.title, "احتياجات")
        m.tick(0.5 + config.SCAN_INTERVAL)               # انتقل للعبارة الثانية
        self.assertEqual(m.current, "أحتاج طعام")
        self.assertEqual(m.select(5.0), "أحتاج طعام")
        self.assertFalse(m.active)

    def test_pause_while_eyes_closed_keeps_highlight(self):
        from scan_menu import ScanMenu
        m = ScanMenu()
        m.open(0.0)
        t = 0.0
        while t < config.SCAN_INTERVAL * 3:              # العين مغلقة طويلاً: لا ينتقل المؤشر
            m.tick(t, paused=True); t += 0.05
        self.assertEqual(m.index, 0)
        m.tick(t + 0.1)
        self.assertEqual(m.index, 1)                      # بعد الفتح ينتقل فوراً (انتهى وقت الخيار)

    def test_back_and_close_items(self):
        from scan_menu import BACK, CLOSE, ScanMenu
        m = ScanMenu()
        m.open(0.0)
        m.select(0.0)
        m.index = m.items().index(BACK)
        self.assertIsNone(m.select(1.0))
        self.assertEqual(m.title, "القائمة")
        m.index = m.items().index(CLOSE)
        m.select(2.0)
        self.assertFalse(m.active)

    def test_auto_close_after_idle_cycles(self):
        from scan_menu import ScanMenu
        m = ScanMenu()
        m.open(0.0)
        t = 0.0
        while m.active and t < 1000:
            t += 0.1
            m.tick(t)
        expected = len(m.items()) * config.SCAN_INTERVAL * config.SCAN_MAX_CYCLES
        self.assertAlmostEqual(t, expected, delta=config.SCAN_INTERVAL)


class CameraUiAndLoggingTests(unittest.TestCase):
    def test_camera_reads_and_reconnects_after_stream_ends(self):
        import tempfile
        import cv2
        import numpy as np
        from camera import Camera
        path = os.path.join(tempfile.mkdtemp(), "clip.mp4")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for i in range(20):
            w.write(np.full((48, 64, 3), i * 10, np.uint8))
        w.release()
        cam = Camera(path)
        try:
            frame, seq = None, -1
            end = time.time() + 6
            while time.time() < end and cam._seq <= 20:      # 20 إطاراً ثم انقطاع ثم إعادة فتح
                f, seq = cam.read(seq)
                frame = f if f is not None else frame
                time.sleep(0.01)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (48, 64, 3))
            self.assertGreater(cam._seq, 20, "لم يُعِد الاتصال بعد انتهاء البث")
        finally:
            cam.release()

    def test_ui_renders_every_state(self):
        import numpy as np
        from ui import H, W, UIState, render
        frame = np.zeros((480, 640, 3), np.uint8)
        now = time.time()
        states = [
            UIState(frame=frame, face=False, calibrating=True, now=now),
            UIState(frame=frame, face=True, calibrating=False, now=now, emotion="sad", emotion_conf=0.6,
                    emotion_probs=np.full(7, 1 / 7), messages=[("10:00:00", "أحتاج ماء", "request")],
                    last_message_at=now),
            UIState(frame=frame, face=True, calibrating=False, pose_ok=False, now=now, emergency_until=now + 5),
            UIState(frame=frame, face=True, calibrating=False, now=now, menu_active=True, menu_title="ألم",
                    menu_items=["ألم في الرأس", "رجوع"], menu_index=1, menu_progress=0.4, ack_until=now + 5),
            UIState(frame=None, now=now, emotion_available=False, blynk_status="error", muted=True),
        ]
        for s in states:
            self.assertEqual(render(s).shape, (H, W, 3))

    def test_old_session_logs_are_deleted(self):
        import tempfile
        from app_logging import cleanup_old_logs
        saved = config.LOG_DIR
        config.LOG_DIR = tempfile.mkdtemp()
        try:
            old, new, other = (os.path.join(config.LOG_DIR, n) for n in
                               ("session_old.csv", "session_new.csv", "app.log"))
            for p in (old, new, other):
                open(p, "w").close()
            os.utime(old, (time.time() - 40 * 86400,) * 2)
            os.utime(other, (time.time() - 40 * 86400,) * 2)
            self.assertEqual(cleanup_old_logs(days=30), 1)
            self.assertFalse(os.path.exists(old))
            self.assertTrue(os.path.exists(new) and os.path.exists(other))
        finally:
            config.LOG_DIR = saved


class EvaluationScoringTests(unittest.TestCase):
    SEGS = [
        {"start": 0, "end": 10, "instruction": "طبيعي", "expected": None},
        {"start": 10, "end": 17, "instruction": "مرتين", "expected": 2},
        {"start": 17, "end": 21, "instruction": "استرح", "expected": None},
        {"start": 21, "end": 27, "instruction": "طوارئ", "expected": "emergency"},
    ]

    def test_hits_and_misses(self):
        from evaluate_blink import score
        hits, expected, fas = score([(15.0, 2)], self.SEGS)
        self.assertEqual(hits, {2: 1})
        self.assertEqual(expected, {2: 1, "emergency": 1})
        self.assertEqual(fas, [])

    def test_message_slightly_after_window_counts(self):
        from evaluate_blink import score
        hits, _, fas = score([(17.0 + config.PATTERN_TIMEOUT + 0.5, 2)], self.SEGS)
        self.assertEqual(hits, {2: 1})
        self.assertEqual(fas, [])

    def test_false_alarm_in_natural_segment_and_wrong_type(self):
        from evaluate_blink import score
        hits, _, fas = score([(5.0, 2), (14.0, 3), (23.0, "emergency")], self.SEGS)
        self.assertEqual(hits, {"emergency": 1})
        self.assertEqual([(t, k) for t, k, _ in fas], [(5.0, 2), (14.0, 3)])

    def test_duplicate_detection_in_same_trial_is_false_alarm(self):
        from evaluate_blink import score
        hits, _, fas = score([(12.0, 2), (16.0, 2)], self.SEGS)
        self.assertEqual(hits, {2: 1})
        self.assertEqual(len(fas), 1)

    def test_protocol_is_well_formed(self):
        from record_protocol import build_segments
        segs = build_segments()
        self.assertTrue(all(a["end"] == b["start"] for a, b in zip(segs, segs[1:])))
        em = [s["start"] for s in segs if s["expected"] == "emergency"]
        self.assertTrue(all(b - a > config.EMERGENCY_COOLDOWN for a, b in zip(em, em[1:])))


class PatientProfileTests(unittest.TestCase):
    KEYS = ("BLINK_MESSAGES", "LONG_BLINK_MESSAGE", "DELIBERATE_BLINK_MIN_SECONDS",
            "LONG_BLINK_SECONDS", "PATTERN_TIMEOUT", "EMERGENCY_COOLDOWN", "BLINK_SIGNAL")

    def setUp(self):
        self.saved = {k: getattr(config, k) for k in self.KEYS}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(config, k, v)

    def write(self, data):
        import json
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, f, ensure_ascii=False)
        f.close()
        return f.name

    def test_default_profile_loads(self):
        from patient_profile import load_profile
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        name = load_profile(os.path.join(here, "patients", "default.json"))
        self.assertTrue(name)
        self.assertEqual(config.BLINK_MESSAGES[2], "أحتاج ماء")

    def test_custom_messages_and_timing_apply(self):
        from patient_profile import load_profile
        load_profile(self.write({"blink_messages": {"2": "نادِ ابني"}, "timing": {"long_blink_seconds": 2.0}}))
        self.assertEqual(config.BLINK_MESSAGES, {2: "نادِ ابني"})
        self.assertEqual(config.LONG_BLINK_SECONDS, 2.0)

    def test_single_blink_pattern_rejected(self):
        from patient_profile import ProfileError, load_profile
        with self.assertRaises(ProfileError):
            load_profile(self.write({"blink_messages": {"1": "x"}}))

    def test_inconsistent_timing_rejected(self):
        from patient_profile import ProfileError, load_profile
        with self.assertRaises(ProfileError):
            load_profile(self.write({"timing": {"deliberate_blink_min_seconds": 2, "long_blink_seconds": 1}}))


class EmotionBaselineTests(unittest.TestCase):
    L = config.EMOTION_LABELS

    def probs(self, **kw):
        import numpy as np
        p = np.full(len(self.L), 0.01)
        for k, v in kw.items():
            p[self.L.index(k)] = v
        return p / p.sum()

    def label(self, p, baseline):
        import numpy as np
        from emotion_detector import baseline_adjust
        return self.L[int(np.argmax(baseline_adjust(p, baseline)))]

    def test_resting_face_that_model_reads_as_sad_becomes_neutral(self):
        rest = self.probs(sad=0.7, angry=0.1, neutral=0.1)
        self.assertEqual(self.label(rest, rest), "neutral")

    def test_real_smile_is_detected_despite_sad_baseline(self):
        rest = self.probs(sad=0.7, angry=0.1, neutral=0.1)
        self.assertEqual(self.label(self.probs(happy=0.8, sad=0.1), rest), "happy")

    def test_real_sadness_above_baseline_is_detected(self):
        rest = self.probs(neutral=0.7, sad=0.1, angry=0.1)
        self.assertEqual(self.label(self.probs(sad=0.8, neutral=0.1), rest), "sad")

    def test_small_noise_around_rest_stays_neutral(self):
        rest = self.probs(sad=0.6, neutral=0.2, angry=0.1)
        self.assertEqual(self.label(self.probs(sad=0.5, neutral=0.25, angry=0.15), rest), "neutral")


class _Recorder(BaseHTTPRequestHandler):
    """خادم Blynk وهمي: يسجّل الطلبات، ويُرجع قيمة زر الاستلام لطلبات get."""
    calls = []
    ack_value = "0"

    def do_GET(self):
        u = urlparse(self.path)
        _Recorder.calls.append((u.path, {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}))
        body = _Recorder.ack_value.encode() if u.path.endswith("/get") else b""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class BlynkTests(unittest.TestCase):
    SETTINGS = ("BLYNK_ENABLED", "BLYNK_TOKEN", "BLYNK_SERVER", "BLYNK_PENDING_FILE",
                "BLYNK_RETRY_BASE_SECONDS", "BLYNK_ACK_POLL_SECONDS")

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.server = HTTPServer(("127.0.0.1", 0), _Recorder)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls._saved = {k: getattr(config, k) for k in cls.SETTINGS}
        cls.live_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.tmp = tempfile.mkdtemp()
        config.BLYNK_ENABLED, config.BLYNK_TOKEN = True, "TESTTOKEN"
        config.BLYNK_RETRY_BASE_SECONDS, config.BLYNK_ACK_POLL_SECONDS = 0.01, 0.05

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            setattr(config, k, v)
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Recorder.calls.clear()
        _Recorder.ack_value = "0"
        config.BLYNK_SERVER = self.live_url
        config.BLYNK_PENDING_FILE = os.path.join(self.tmp, f"pending_{self._testMethodName}.jsonl")
        self.clients = []

    def tearDown(self):
        for c in self.clients:
            c.close()
        time.sleep(0.1)

    def make(self):
        from blynk_client import Blynk
        b = Blynk()
        self.clients.append(b)
        return b

    def paths(self):
        return [p for p, _ in _Recorder.calls]

    def wait_for(self, n, timeout=5):
        end = time.time() + timeout
        while len(_Recorder.calls) < n and time.time() < end:
            time.sleep(0.05)

    def test_write_and_alert_requests(self):
        b = self.make()
        b.write("V0", "أحتاج ماء", min_interval=0)
        b.alert("طوارئ", force=True)
        self.wait_for(2)
        paths = {p: q for p, q in _Recorder.calls}
        self.assertEqual(paths["/external/api/update"], {"token": "TESTTOKEN", "V0": "أحتاج ماء"})
        self.assertEqual(paths["/external/api/logEvent"],
                         {"token": "TESTTOKEN", "code": config.BLYNK_EVENT_CODE, "description": "طوارئ"})
        self.assertEqual(b.status, "ok")

    def test_alert_rate_limit(self):
        b = self.make()
        b.alert("a"); b.alert("b"); b.alert("c")
        self.wait_for(1)
        time.sleep(0.3)
        self.assertEqual(self.paths().count("/external/api/logEvent"), 1)

    def test_disabled_without_token(self):
        config.BLYNK_TOKEN = ""
        try:
            b = self.make()
            b.alert("x", force=True)
            time.sleep(0.3)
            self.assertFalse(b.enabled)
            self.assertEqual(_Recorder.calls, [])
        finally:
            config.BLYNK_TOKEN = "TESTTOKEN"

    def test_caregiver_ack_is_received_and_button_reset(self):
        b = self.make()
        b.alert("أحتاج ماء")
        _Recorder.ack_value = "1"
        end = time.time() + 3
        while not b.consume_ack() and time.time() < end:
            time.sleep(0.02)
        self.assertLess(time.time(), end, "لم يصل تأكيد الاستلام")
        gets = [q for p, q in _Recorder.calls if p == "/external/api/get"]
        self.assertIn(config.BLYNK_PIN_ACK, gets[0])          # صيغة get?token=..&V4
        time.sleep(0.2)
        self.assertIn(("/external/api/update", {"token": "TESTTOKEN", config.BLYNK_PIN_ACK: "0"}), _Recorder.calls)
        self.assertFalse(b.consume_ack())                     # مرة واحدة فقط

    def test_no_ack_polling_when_nothing_awaits(self):
        self.make()
        time.sleep(0.4)                                       # 8 فترات فحص
        self.assertNotIn("/external/api/get", self.paths())

    def test_heartbeat_rate_limited(self):
        b = self.make()
        for _ in range(5):
            b.heartbeat()
        time.sleep(0.3)
        beats = [q for p, q in _Recorder.calls if config.BLYNK_PIN_HEARTBEAT in q]
        self.assertEqual(len(beats), 1)

    def test_offline_messages_are_saved_and_resent(self):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); dead_port = s.getsockname()[1]; s.close()
        config.BLYNK_SERVER = f"http://127.0.0.1:{dead_port}"      # الإنترنت مقطوع
        b = self.make()
        b.alert("طوارئ!", force=True)
        end = time.time() + 20          # ويندوز يستغرق ~2 ث لرفض كل محاولة اتصال بمنفذ مغلق
        while not os.path.exists(config.BLYNK_PENDING_FILE) and time.time() < end:
            time.sleep(0.05)
        self.assertTrue(os.path.exists(config.BLYNK_PENDING_FILE), "لم تُحفظ الرسالة")
        self.assertEqual(b.status, "error")

        config.BLYNK_SERVER = self.live_url                   # عاد الاتصال
        b.heartbeat()                                         # أي طلب ناجح يُطلق إعادة الإرسال
        end = time.time() + 5
        while "/external/api/logEvent" not in self.paths() and time.time() < end:
            time.sleep(0.05)
        events = [q for p, q in _Recorder.calls if p == "/external/api/logEvent"]
        self.assertEqual(events[0]["description"], "طوارئ!")
        self.assertFalse(os.path.exists(config.BLYNK_PENDING_FILE))


if __name__ == "__main__":
    unittest.main()
