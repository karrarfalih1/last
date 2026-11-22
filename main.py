"""
AI-Based Blink and Emotion Detection System for Assistive Communication

التشغيل:
    python main.py                                   # كاميرا اللابتوب
    python main.py --source http://192.168.1.5:8080/video   # كاميرا الهاتف (IP Webcam)
المفاتيح:  Q خروج | R إعادة معايرة | M كتم الصوت | S لقطة شاشة
"""
import argparse
import os
import time

import cv2

import config
import patient_profile
from alerts import LocalAlert
from app_logging import cleanup_old_logs, setup as setup_logging
from blink_detector import LEFT_EYE, RIGHT_EYE
from blynk_client import Blynk
from camera import Camera
from emotion_detector import EmotionDetector
from event_log import EventLog
from face_tracker import FaceTracker
from patient_profile import load_profile
from pipeline import BlinkPipeline
from scan_menu import ScanMenu
from ui import UIState, render

logger = setup_logging()

WINDOW = "Assistive Communication  -  Blink & Emotion Detection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="مصدر الكاميرا: رقم (0) أو رابط بث الهاتف")
    p.add_argument("--duration", type=float, help="إنهاء تلقائي بعد N ثانية (للاختبار)")
    p.add_argument("--snapshot", help="حفظ لقطة نهائية في هذا المسار عند الخروج (للاختبار)")
    p.add_argument("--profile", default="patients/default.json", help="ملف المريض (JSON)")
    return p.parse_args()


def main():
    args = parse_args()
    removed = cleanup_old_logs()
    if removed:
        logger.info(f"حُذف {removed} سجل أقدم من {config.LOG_RETENTION_DAYS} يوماً")
    patient = ""
    if args.profile and os.path.exists(args.profile):
        patient = load_profile(args.profile)
        logger.info(f"ملف المريض: {args.profile}  {patient}")
    elif args.profile:
        logger.warning(f"لم يُعثر على {args.profile} — الإعدادات الافتراضية")
    source = config.CAMERA_SOURCE
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source

    blynk, alert, log = Blynk(), LocalAlert(), EventLog(config.LOG_DIR)
    emotions = EmotionDetector()
    pipeline = BlinkPipeline()
    blinks = pipeline.blinks
    menu = ScanMenu(patient_profile.CUSTOM_MENU)
    fullscreen = False
    logger.info(f"Blynk: {'مفعّل' if blynk.enabled else 'غير مفعّل (ضع BLYNK_TOKEN في ملف .env)'}"
                f"   |   إشارة الرمش: {pipeline.signal}   |   سجل الجلسة: {log.path}")
    log.log("start", f"source={source} signal={pipeline.signal} model={config.EMOTION_MODEL_PATH} "
                     f"profile={args.profile} patient={patient}")

    cam = Camera(source)
    tracker = FaceTracker()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # قابلة للتكبير وملء الشاشة
    cv2.resizeWindow(WINDOW, 1040, 620)

    state = UIState(emotion_available=emotions.available)
    seq, frame_i, fps_t, fps_n = -1, 0, time.time(), 0
    started = time.time()
    last_face_time = started
    no_face_alerted = False
    bad_emotion_since = None
    last_emotion_logged = None
    last_emotion_sent, last_emotion_sent_at = None, -1e9
    was_calibrating = False
    last_frame_ui = None

    def emit(msg_text, kind):
        """رسالة جديدة: واجهة + صوت + Blynk + سجل."""
        now = time.time()
        state.messages.append((time.strftime("%H:%M:%S"), msg_text, kind))
        state.last_message_at = now
        if kind == "emergency":
            state.emergency_until = now + 6
        logger.info(f"[{kind}] {msg_text}")
        log.log(kind, msg_text)
        alert.play(kind)
        if kind in ("request", "emergency"):
            alert.say(msg_text)            # نطق الرسالة بصوت مسموع (صوت المريض)
        blynk.write(config.BLYNK_PIN_MESSAGE, msg_text, min_interval=0, persist=True)
        blynk.alert(msg_text, force=(kind == "emergency"))
        state.ack_until = 0.0

    def handle(msg, now):
        """رسالة من خط الرمش: طلب/طوارئ، أو فتح القائمة، أو اختيار داخلها."""
        if msg.kind == "menu":
            menu.open(now)
            blinks.set_mode("select")
            alert.play("menu")
            log.log("menu", "open")
        elif msg.kind == "select":
            alert.play("select")
            phrase = menu.select(now)
            if phrase:
                emit(phrase, "request")
            if not menu.active:
                blinks.set_mode("pattern")
        else:
            if msg.kind == "emergency" and menu.active:
                menu.close()
                blinks.set_mode("pattern")
            emit(msg.text, msg.kind)

    try:
        while True:
            blynk.heartbeat()
            if blynk.consume_ack():       # مقدّم الرعاية ضغط "تم الاستلام"
                state.ack_until = time.time() + config.ACK_DISPLAY_SECONDS
                state.messages.append((time.strftime("%H:%M:%S"), "تم الاستلام — المساعد في الطريق", "ack"))
                log.log("ack", "caregiver acknowledged")
                alert.play("ack")
                alert.say(config.ACK_PHRASE)
            frame, seq = cam.read(seq)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (args.duration and time.time() - started > args.duration):
                break
            if key == ord("r"):
                pipeline.reset(); blinks = pipeline.blinks; emotions.reset(); menu.close()
                state.emotion, state.emotion_probs, bad_emotion_since = None, None, None
                log.log("calibration", "restart")
            if key == ord("m"):
                alert.toggle_mute()
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("s") and last_frame_ui is not None:
                os.makedirs(config.LOG_DIR, exist_ok=True)
                path = os.path.join(config.LOG_DIR, time.strftime("shot_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(path, last_frame_ui); logger.info(f"لقطة: {path}")
            if frame_i > 0 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break    # أُغلقت النافذة بزر X

            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            frame_i += 1
            fps_n += 1
            if now - fps_t >= 1:
                state.fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            obs = tracker.process(frame)
            state.now = now
            state.blynk_status = blynk.status
            state.muted = alert.muted

            # ---- الرمش (معايرة + زاوية الرأس + الحالة + النمط) ----
            msg = pipeline.step(obs, now)
            if msg:
                handle(msg, now)
            was_menu = menu.active
            menu.tick(now, paused=blinks.closed_since is not None)
            if was_menu and not menu.active:          # إغلاق تلقائي بعد دورات بلا اختيار
                blinks.set_mode("pattern")
                log.log("menu", "auto-close")
            state.menu_active = menu.active
            if menu.active:
                state.menu_title, state.menu_items = menu.title, menu.items()
                state.menu_index, state.menu_progress = menu.index, menu.progress(now)

            if obs is not None:
                pts = obs.points
                state.face = True
                last_face_time, no_face_alerted, state.no_face_for = now, False, 0.0
                eye = pipeline.eye
                state.ear, state.closed, state.pose_ok = pipeline.value, pipeline.closed, pipeline.pose_ok
                state.calibrating, state.calib_remaining = eye.calibrating, eye.calibration_remaining
                state.close_thr, state.open_thr = eye.close_threshold, eye.open_threshold
                state.signal_max = (eye.baseline or pipeline.value or 1.0) * 1.4
                state.yaw, state.pitch = obs.yaw, obs.pitch
                state.calib_warning = pipeline.calib_warning
                state.calib_note = pipeline.calib_note
                state.eye_points = [tuple(pts[i]) for i in LEFT_EYE + RIGHT_EYE]

                # ---- المعايرة الشخصية للمشاعر: وجه هادئ في أول ثوانٍ ----
                if pipeline.calibrating:
                    was_calibrating = True
                    if frame_i % config.EMOTION_EVERY_N_FRAMES == 0:
                        emotions.collect_baseline(frame, pts)
                elif was_calibrating:
                    was_calibrating = False
                    emotions.finish_baseline()
                    log.log("calibration", f"{pipeline.signal}_baseline={eye.baseline:.3f} "
                                           f"yaw0={pipeline.pose.yaw0:.1f} pitch0={pipeline.pose.pitch0:.1f}")
                    if pipeline.calib_note:
                        logger.warning(pipeline.calib_note)

                # ---- المشاعر (بعد المعايرة فقط، وكل N إطار) ----
                if not pipeline.calibrating and frame_i % config.EMOTION_EVERY_N_FRAMES == 0:
                    res = emotions.predict(frame, pts)
                    if res:
                        label, conf, probs = res
                        state.emotion, state.emotion_conf, state.emotion_probs = label, conf, probs
                        # إلى Blynk: عند التغيّر فقط، وبحد أقصى مرة كل BLYNK_EMOTION_MIN_INTERVAL
                        if (label != last_emotion_sent and conf >= 0.5
                                and now - last_emotion_sent_at >= config.BLYNK_EMOTION_MIN_INTERVAL):
                            blynk.write(config.BLYNK_PIN_EMOTION, config.EMOTION_LABELS_AR[label], min_interval=0)
                            last_emotion_sent, last_emotion_sent_at = label, now
                        if label != last_emotion_logged and conf >= 0.5:
                            log.log("emotion", f"{label} ({conf:.0%})")
                            last_emotion_logged = label
                        # تنبيه إذا استمرت مشاعر سلبية بثقة كافية
                        if label in config.EMOTION_ALERT and conf >= config.EMOTION_ALERT_MIN_CONF:
                            bad_emotion_since = bad_emotion_since or now
                            if now - bad_emotion_since > config.EMOTION_ALERT_SECONDS:
                                emit(f"المريض يبدو {config.EMOTION_LABELS_AR[label]}", "emotion")
                                bad_emotion_since = None
                        else:
                            bad_emotion_since = None
            else:
                state.face = False
                state.eye_points = []
                state.no_face_for = now - last_face_time
                if state.no_face_for > config.NO_FACE_ALERT_SECONDS and not no_face_alerted:
                    no_face_alerted = True
                    emit("المريض غير ظاهر في الكاميرا", "warning")

            state.frame = frame
            state.blink_count = blinks.blink_count
            state.pattern_left = blinks.pattern_time_left(now)
            state.closed_for = blinks.closed_for(now)
            state.ignored_blinks = blinks.ignored_blinks

            last_frame_ui = render(state)
            cv2.imshow(WINDOW, last_frame_ui)
    finally:
        if args.snapshot and last_frame_ui is not None:
            cv2.imwrite(args.snapshot, last_frame_ui)
        log.log("stop", f"frames={frame_i}")
        log.close()
        blynk.close()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
