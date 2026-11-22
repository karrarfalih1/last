# ===================== إعدادات المشروع =====================

# ---------------- الكاميرا ----------------
# 0 = كاميرا اللابتوب / USB      "http://192.168.1.5:8080/video" = كاميرا الهاتف (IP Webcam)
# يمكن تجاوزها من سطر الأوامر:  python main.py --source http://192.168.1.5:8080/video
CAMERA_SOURCE = 0

# ---------------- Blynk ----------------
# التوكن سرّي: لا تكتبه هنا. ضعه في ملف .env بجانب المشروع بهذا الشكل:
#     BLYNK_TOKEN=xxxxxxxxxxxxxxxx
#     BLYNK_SERVER=https://fra1.blynk.cloud
# أو كمتغير بيئة بنفس الاسم. يتفعّل Blynk تلقائياً عند وجود التوكن.
import os as _os


def _load_env_file(path=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env")):
    if _os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    _os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()
BLYNK_TOKEN = _os.environ.get("BLYNK_TOKEN", "")
BLYNK_SERVER = _os.environ.get("BLYNK_SERVER", "https://blynk.cloud")   # حسب منطقتك: fra1 / ny3 / blr1 ...
BLYNK_ENABLED = bool(BLYNK_TOKEN)
BLYNK_PIN_MESSAGE = "V0"              # String: رسالة المريض
BLYNK_PIN_EMOTION = "V1"              # String: المشاعر الحالية
BLYNK_PIN_HEARTBEAT = "V2"            # String: وقت آخر نبض (النظام يعمل)
BLYNK_PIN_ACK = "V4"                  # Integer 0/1: زر "تم الاستلام" لمقدّم الرعاية (Switch mode)
BLYNK_EVENT_CODE = "patient_alert"    # كود الـ Event في Blynk (إشعار للهاتف)
BLYNK_ALERT_MIN_INTERVAL = 5          # أقل فاصل (ثوانٍ) بين إشعارين (Blynk تحدّ عدد الأحداث)
# الخطة المجانية في Blynk تحدّ عدد الرسائل الشهرية، لذلك نقتصد:
BLYNK_HEARTBEAT_SECONDS = 60          # نبض كل دقيقة
BLYNK_ACK_POLL_SECONDS = 2            # قراءة زر الاستلام كل ثانيتين…
BLYNK_ACK_WAIT_SECONDS = 600          # …فقط خلال 10 دقائق بعد آخر رسالة لم تُؤكَّد
BLYNK_EMOTION_MIN_INTERVAL = 10       # المشاعر تُرسل عند تغيّرها فقط، وبحد أقصى مرة كل 10 ث
BLYNK_RETRY_BASE_SECONDS = 1.0        # الإعادة بعد 1، 2، 4 ثوانٍ
BLYNK_PENDING_FILE = "logs/blynk_pending.jsonl"   # رسائل مهمة لم تُرسل (تُعاد عند عودة الإنترنت)
ACK_DISPLAY_SECONDS = 12              # مدة عرض "تم الاستلام" للمريض

# ---------------- كشف الرمش ----------------
FACE_LANDMARKER_PATH = "models/face_landmarker.task"
# الإشارة المستخدمة لقياس انفتاح العين:
#   "ear"        = هندسة النقاط فقط (الطريقة الكلاسيكية)
#   "blendshape" = احتمال الإغلاق من شبكة MediaPipe
#   "fused"      = دمج الاثنين (الافتراضي — أكثر متانة)
BLINK_SIGNAL = "fused"
MIN_CALIBRATION_BASELINE = 0.75  # أقل انفتاح مقبول للمعايرة (الإشارة المدمجة/الشبكة ≈ 1.0 والعين مفتوحة)
WARN_CALIBRATION_YAW = 40        # تحذير (غير مانع) إن كان الوجه مائلاً عن الكاميرا أكثر من هذا عند المعايرة
MAX_YAW_DEVIATION = 25       # درجات التفات يميناً/يساراً عن وضع المعايرة قبل تجاهل الرمش
MAX_PITCH_DEVIATION = 20     # درجات ميل للأعلى/الأسفل
ADAPTIVE_BASELINE_RATE = 0.002   # سرعة تكيّف الحد مع تغيّر الإضاءة/التعب (0 = إيقاف)
EAR_THRESHOLD = 0.21         # قيمة احتياطية فقط — الحد الفعلي يُعاير تلقائياً
CALIBRATION_SECONDS = 3      # مدة المعايرة (انظر للكاميرا وعيناك مفتوحتان)
CALIBRATION_CLOSE_RATIO = 0.75   # عين مغلقة إذا نزل EAR تحت 75% من انفتاحك الطبيعي
CALIBRATION_OPEN_RATIO = 0.85    # وتعتبر مفتوحة مجدداً فوق 85% (فرق بين الحدّين يمنع الاهتزاز)

# الرمش اللاإرادي يستغرق ~0.1–0.15 ث، أما المتعمّد فأطول. نتجاهل القصير لتقليل الإنذارات الكاذبة.
DELIBERATE_BLINK_MIN_SECONDS = 0.20
LONG_BLINK_SECONDS = 1.2     # إغلاق أطول من هذا = طوارئ
PATTERN_TIMEOUT = 1.5        # ثوانٍ بعد آخر رمشة لإنهاء النمط وتفسيره
EMERGENCY_COOLDOWN = 10      # ثوانٍ لا تتكرر خلالها رسالة الطوارئ

# معنى كل نمط (عدد الرمشات المتعمّدة -> الرسالة). رمشة واحدة تُتجاهل.
# (تُستبدل من ملف المريض patients/*.json إن وُجد)
BLINK_MESSAGES = {
    2: "أحتاج ماء",
    3: "أحتاج طعام",
    4: "أشعر بألم",
}
LONG_BLINK_MESSAGE = "طوارئ! أحتاج مساعدة فوراً"
MENU_PATTERN = 5             # عدد الرمشات لفتح قائمة المسح (عشرات العبارات)

# ---------------- قائمة المسح (Scanning Menu) ----------------
SCAN_INTERVAL = 2.0          # ثوانٍ على كل خيار قبل الانتقال للتالي
SCAN_MAX_CYCLES = 3          # إغلاق تلقائي بعد 3 دورات كاملة بدون اختيار

# ---------------- الصوت ----------------
AUDIO_DIR = "assets/audio"
AUDIO_VOICE = "ar-IQ-BasselNeural"   # صوت عراقي (edge-tts). بدائل: ar-IQ-RanaNeural, ar-SA-HamedNeural
SPEAK_MESSAGES = True        # نطق رسالة المريض بصوت مسموع
ACK_PHRASE = "تم الاستلام، المساعد في طريقه إليك"

# ---------------- المشاعر ----------------
EMOTION_MODEL_PATH = "emotion_model.keras"
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_LABELS_AR = {
    "angry": "غاضب", "disgust": "منزعج", "fear": "خائف", "happy": "سعيد",
    "neutral": "طبيعي", "sad": "حزين", "surprise": "متفاجئ",
}
EMOTION_EVERY_N_FRAMES = 3         # تحليل المشاعر كل N إطار
EMOTION_SMOOTHING = 8              # متوسط آخر N توقعات لتثبيت النتيجة
EMOTION_FACE_MARGIN = 0.05         # هامش حول الوجه عند القص
EMOTION_NEUTRAL_BIAS = 1.2         # تفضيل "طبيعي" بعد المعايرة الشخصية (أكبر = أقل حساسية للانفعال)
EMOTION_ALERT = {"sad", "fear", "angry"}   # مشاعر تُنبّه إذا استمرت
EMOTION_ALERT_SECONDS = 10         # مدة الاستمرار قبل التنبيه
EMOTION_ALERT_MIN_CONF = 0.45      # لا ننبّه إلا إذا كانت الثقة كافية

# ---------------- السلامة ----------------
NO_FACE_ALERT_SECONDS = 15         # تنبيه إذا اختفى المريض عن الكاميرا
LOCAL_SOUND = True                 # صوت تنبيه محلي (صفارة) على الكمبيوتر
LOG_DIR = "logs"                   # سجل الأحداث (CSV) لكل جلسة
LOG_RETENTION_DAYS = 30            # حذف سجلات الجلسات الأقدم تلقائياً (0 = عدم الحذف)
