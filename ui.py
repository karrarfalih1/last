"""لوحة العرض: الفيديو + لوحة حالة جانبية + لافتة الرسالة الأخيرة. النص العربي يُرسم بـ PIL (cv2.putText لا يدعمه)."""
import re
import time
from dataclasses import dataclass, field

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

import config

W, H = 1040, 620
VIDEO_W, VIDEO_H = 640, 480
FOOTER_H = 36
PANEL_X = VIDEO_W + 16
PANEL_R = W - 16

BG = (18, 20, 26)
CARD = (29, 32, 42)
TEXT = (232, 234, 240)
MUTED = (140, 146, 162)
GREEN = (52, 211, 153)
RED = (248, 81, 73)
AMBER = (245, 180, 60)
BLUE = (96, 165, 250)
PURPLE = (192, 132, 252)

KIND_COLORS = {"emergency": RED, "request": GREEN, "ack": BLUE, "warning": AMBER, "emotion": PURPLE}

EMOTION_COLORS = {"angry": RED, "disgust": (132, 204, 22), "fear": PURPLE, "happy": GREEN,
                  "neutral": MUTED, "sad": BLUE, "surprise": AMBER}

_FONT_FILES = {False: ["segoeui.ttf", "tahoma.ttf", "arial.ttf"], True: ["segoeuib.ttf", "tahomabd.ttf", "arialbd.ttf"]}
_fonts = {}
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        for name in _FONT_FILES[bold]:
            try:
                _fonts[key] = ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
                break
            except OSError:
                continue
        else:
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]


def shape(s):
    return get_display(arabic_reshaper.reshape(s)) if _ARABIC.search(s) else s


def text(d, xy, s, size=14, fill=TEXT, bold=False, anchor="la"):
    d.text(xy, shape(s), font=font(size, bold), fill=fill, anchor=anchor)


def bar(d, x1, y, x2, h, frac, color, track=(48, 52, 66)):
    d.rounded_rectangle((x1, y, x2, y + h), radius=h // 2, fill=track)
    frac = max(0.0, min(1.0, frac))
    if frac > 0.01:
        d.rounded_rectangle((x1, y, x1 + (x2 - x1) * frac, y + h), radius=h // 2, fill=color)


def card(d, y1, y2):
    d.rounded_rectangle((PANEL_X, y1, PANEL_R, y2), radius=10, fill=CARD)


def letterbox(frame, w, h):
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    resized = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return out


@dataclass
class UIState:
    frame: np.ndarray = None
    eye_points: list = field(default_factory=list)   # نقاط العين بالبكسل لرسمها على الفيديو
    face: bool = False
    calibrating: bool = True
    calib_remaining: float = 0.0
    ear: float = 0.0
    closed: bool = False
    close_thr: float = 0.0
    open_thr: float = 0.0
    signal_max: float = 0.5        # أقصى قيمة لشريط الانفتاح
    pose_ok: bool = True
    calib_warning: str = ""
    calib_note: str = ""
    yaw: float = 0.0
    pitch: float = 0.0
    blink_count: int = 0
    pattern_left: float = 0.0
    closed_for: float = 0.0
    ignored_blinks: int = 0
    emotion: str = None
    emotion_conf: float = 0.0
    emotion_probs: np.ndarray = None
    emotion_available: bool = True
    messages: list = field(default_factory=list)      # [(وقت نصي، نص، kind)] الأحدث آخراً
    last_message_at: float = 0.0
    emergency_until: float = 0.0
    ack_until: float = 0.0
    menu_active: bool = False
    menu_title: str = ""
    menu_items: list = field(default_factory=list)
    menu_index: int = 0
    menu_progress: float = 0.0
    fps: float = 0.0
    blynk_status: str = "off"
    muted: bool = False
    no_face_for: float = 0.0
    now: float = field(default_factory=time.time)


def draw_menu(d, s):
    """قائمة المسح فوق الفيديو: الخيار المضاء كبير وواضح للمريض."""
    d.rectangle((0, 0, VIDEO_W, VIDEO_H), fill=(10, 12, 18, 225))
    text(d, (VIDEO_W - 24, 18), s.menu_title, 22, AMBER, True, "ra")
    text(d, (24, 24), "ارمش مرة عند الخيار المطلوب", 14, MUTED, False, "la")
    n = len(s.menu_items)
    row_h = min(52, (VIDEO_H - 70) // max(n, 1))
    y0 = 60
    for i, item in enumerate(s.menu_items):
        y = y0 + i * row_h
        cur = i == s.menu_index
        if cur:
            d.rounded_rectangle((20, y, VIDEO_W - 20, y + row_h - 6), radius=10, fill=(245, 180, 60, 235))
            bar(d, 32, y + row_h - 14, VIDEO_W - 32, 4, s.menu_progress, BG, track=(200, 140, 40))
        special = item in ("رجوع", "إغلاق القائمة")
        size = min(30, row_h - 18) if cur else min(22, row_h - 22)
        col = BG if cur else (MUTED if special else TEXT)
        text(d, (VIDEO_W - 40, y + (row_h - 6) // 2 - (3 if cur else 0)), item, size, col, cur, "rm")


def render(s: UIState):
    canvas = np.full((H, W, 3), BG, np.uint8)

    # ---- الفيديو ----
    if s.frame is not None:
        frame = s.frame.copy()
        fh, fw = frame.shape[:2]
        for (x, y) in s.eye_points:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 120) if not s.closed else (0, 80, 255), -1)
        canvas[:VIDEO_H, :VIDEO_W] = letterbox(frame, VIDEO_W, VIDEO_H)

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- تراكب الفيديو ----
    if s.calibrating:
        d.rectangle((0, 0, VIDEO_W, VIDEO_H), fill=(0, 0, 0, 150))
        text(d, (VIDEO_W // 2, VIDEO_H // 2 - 34), "معايرة… انظر للكاميرا بوجه هادئ", 22, AMBER, True, "mm")
        text(d, (VIDEO_W // 2, VIDEO_H // 2 - 4), "عيناك مفتوحتان، بدون ابتسام أو عبوس", 15, TEXT, False, "mm")
        if s.face:
            text(d, (VIDEO_W // 2, VIDEO_H // 2 + 30), f"{max(s.calib_remaining, 0.5):.0f}", 34, TEXT, True, "mm")
        else:
            text(d, (VIDEO_W // 2, VIDEO_H // 2 + 30), "بانتظار ظهور الوجه…", 18, RED, True, "mm")
        if s.calib_warning:
            text(d, (VIDEO_W // 2, VIDEO_H // 2 + 80), s.calib_warning, 16, RED, True, "mm")
    elif not s.face:
        d.rectangle((0, 0, VIDEO_W, 44), fill=(160, 30, 30, 200))
        text(d, (VIDEO_W // 2, 22), "لا يوجد وجه في الكاميرا", 18, TEXT, True, "mm")
    elif not s.pose_ok:
        d.rectangle((0, 0, VIDEO_W, 44), fill=(150, 100, 10, 200))
        text(d, (VIDEO_W // 2, 22), "الرأس مائل — كشف الرمش متوقف مؤقتاً", 18, TEXT, True, "mm")
    elif s.calib_note:
        d.rectangle((0, 0, VIDEO_W, 30), fill=(150, 100, 10, 170))
        text(d, (VIDEO_W // 2, 15), s.calib_note, 14, TEXT, False, "mm")
    if s.menu_active and not s.calibrating:
        draw_menu(d, s)
    if s.now < s.emergency_until and int(s.now * 3) % 2 == 0:
        d.rectangle((0, 0, VIDEO_W - 1, VIDEO_H - 1), outline=RED, width=10)

    # ---- لافتة الرسالة الأخيرة (تحت الفيديو) ----
    by1, by2 = VIDEO_H + 8, H - FOOTER_H - 8
    recent = s.messages and s.now - s.last_message_at < 10
    if s.now < s.ack_until:
        d.rounded_rectangle((8, by1, VIDEO_W - 8, by2), radius=10, fill=(20, 40, 70), outline=BLUE, width=2)
        cy = (by1 + by2) // 2 - 12
        text(d, (VIDEO_W // 2 - 14, cy), "تم الاستلام", 30, BLUE, True, "mm")
        cx = VIDEO_W // 2 + 82                     # علامة صح مرسومة (الخط لا يحتوي الرمز ✓)
        d.line([(cx, cy), (cx + 8, cy + 9), (cx + 24, cy - 10)], fill=BLUE, width=5, joint="curve")
        text(d, (VIDEO_W // 2, (by1 + by2) // 2 + 22), "المساعد في الطريق إليك", 18, TEXT, False, "mm")
    elif recent:
        t, msg, kind = s.messages[-1]
        col = KIND_COLORS.get(kind, GREEN)
        d.rounded_rectangle((8, by1, VIDEO_W - 8, by2), radius=10, fill=(col[0] // 4, col[1] // 4, col[2] // 4), outline=col, width=2)
        text(d, (VIDEO_W // 2, (by1 + by2) // 2 - 6), msg, 32, col, True, "mm")
        text(d, (VIDEO_W - 22, by2 - 8), t, 12, MUTED, False, "rd")
    else:
        d.rounded_rectangle((8, by1, VIDEO_W - 8, by2), radius=10, fill=CARD)
        text(d, (VIDEO_W // 2, by1 + 26), "بانتظار رسالة من المريض…", 20, MUTED, True, "mm")
        legend = " · ".join(f"{n}: {m}" for n, m in config.BLINK_MESSAGES.items())
        if config.MENU_PATTERN:
            legend += f" · {config.MENU_PATTERN}: القائمة"
        text(d, (VIDEO_W // 2, by1 + 58), legend, 13, MUTED, False, "mm")
        text(d, (VIDEO_W // 2, by1 + 80), f"إغلاق العين {config.LONG_BLINK_SECONDS:g} ثانية = طوارئ", 13, RED, False, "mm")

    # ---- رأس اللوحة ----
    text(d, (PANEL_R, 14), "نظام التواصل المساعد", 20, TEXT, True, "ra")
    if s.calibrating:
        chip, chip_col = "معايرة", AMBER
    elif not s.face:
        chip, chip_col = "لا وجه", RED
    elif not s.pose_ok:
        chip, chip_col = "الرأس مائل", AMBER
    else:
        chip, chip_col = "مراقبة", GREEN
    d.rounded_rectangle((PANEL_X, 12, PANEL_X + 96, 38), radius=13, fill=(chip_col[0] // 4, chip_col[1] // 4, chip_col[2] // 4))
    d.ellipse((PANEL_X + 10, 21, PANEL_X + 18, 29), fill=chip_col)
    text(d, (PANEL_X + 26, 25), chip, 13, chip_col, True, "lm")

    # ---- بطاقة العين ----
    y1, y2 = 52, 138
    card(d, y1, y2)
    text(d, (PANEL_R - 12, y1 + 10), "العين", 15, MUTED, True, "ra")
    if s.calibrating:
        st, st_col = "معايرة", AMBER
    elif not s.face:
        st, st_col = "—", MUTED
    else:
        st, st_col = ("مغلقة", RED) if s.closed else ("مفتوحة", GREEN)
    text(d, (PANEL_X + 12, y1 + 8), st, 20, st_col, True, "la")
    bx1, bx2 = PANEL_X + 12, PANEL_R - 12
    scale = max(s.signal_max, 1e-6)
    bar(d, bx1, y1 + 46, bx2, 10, s.ear / scale, st_col if s.face else MUTED)
    if not s.calibrating:
        for thr, col in ((s.close_thr, RED), (s.open_thr, GREEN)):
            x = bx1 + (bx2 - bx1) * min(thr / scale, 1)
            d.line((x, y1 + 42, x, y1 + 60), fill=col, width=2)
    text(d, (bx1, y1 + 66), f"انفتاح {s.ear:.2f}" + ("" if s.calibrating else f"  حد {s.close_thr:.2f}"), 12, MUTED)
    text(d, (bx2, y1 + 66), f"yaw {s.yaw:+.0f}°  pitch {s.pitch:+.0f}°", 12, AMBER if not s.pose_ok else MUTED, False, "ra")

    # ---- بطاقة الرمش ----
    y1, y2 = 146, 240
    card(d, y1, y2)
    text(d, (PANEL_R - 12, y1 + 10), "الرمشات المتعمدة", 15, MUTED, True, "ra")
    for i in range(5):
        cx = PANEL_X + 22 + i * 30
        filled = i < s.blink_count
        d.ellipse((cx - 10, y1 + 12, cx + 10, y1 + 32), fill=AMBER if filled else (48, 52, 66))
        text(d, (cx, y1 + 22), str(i + 1), 12, BG if filled else MUTED, True, "mm")
    text(d, (PANEL_X + 12, y1 + 42), "نهاية النمط", 11, MUTED)
    bar(d, PANEL_X + 90, y1 + 46, PANEL_R - 12, 7, s.pattern_left / config.PATTERN_TIMEOUT, AMBER)
    prog = s.closed_for / config.LONG_BLINK_SECONDS
    text(d, (PANEL_X + 12, y1 + 60), "نحو الطوارئ", 11, MUTED)
    bar(d, PANEL_X + 90, y1 + 64, PANEL_R - 12, 7, prog if s.closed_for > config.DELIBERATE_BLINK_MIN_SECONDS else 0,
        RED if prog > 0.7 else AMBER)
    text(d, (PANEL_X + 12, y1 + 78), f"رمشات لاإرادية تم تجاهلها: {s.ignored_blinks}", 11, MUTED)

    # ---- بطاقة المشاعر ----
    y1, y2 = 248, 420
    card(d, y1, y2)
    text(d, (PANEL_R - 12, y1 + 10), "المشاعر", 15, MUTED, True, "ra")
    if s.emotion:
        col = EMOTION_COLORS[s.emotion]
        text(d, (PANEL_X + 12, y1 + 6), config.EMOTION_LABELS_AR[s.emotion], 22, col, True)
        text(d, (PANEL_X + 110, y1 + 14), f"{s.emotion_conf:.0%}", 14, MUTED)
        for i, lab in enumerate(config.EMOTION_LABELS):
            ry = y1 + 40 + i * 18
            text(d, (PANEL_R - 12, ry), config.EMOTION_LABELS_AR[lab], 12, TEXT if lab == s.emotion else MUTED, False, "ra")
            p = float(s.emotion_probs[i]) if s.emotion_probs is not None else 0
            bar(d, PANEL_X + 12, ry + 4, PANEL_R - 72, 8, p, EMOTION_COLORS[lab])
    else:
        msg = ("نموذج المشاعر غير محمّل" if not s.emotion_available
               else "بعد المعايرة…" if s.calibrating else "بانتظار الوجه…")
        text(d, (PANEL_X + 12, y1 + 40), msg, 14, MUTED)

    # ---- سجل الرسائل ----
    y1, y2 = 428, H - FOOTER_H - 8
    card(d, y1, y2)
    text(d, (PANEL_R - 12, y1 + 10), "سجل الرسائل", 15, MUTED, True, "ra")
    if not s.messages:
        text(d, (PANEL_X + 12, y1 + 40), "لا رسائل بعد", 13, MUTED)
    for i, (t, msg, kind) in enumerate(reversed(s.messages[-5:])):
        ry = y1 + 36 + i * 22
        text(d, (PANEL_X + 12, ry), t, 12, MUTED)
        text(d, (PANEL_R - 12, ry - 2), msg, 14, KIND_COLORS.get(kind, TEXT) if kind != "request" else TEXT,
             kind == "emergency", "ra")

    # ---- الشريط السفلي ----
    d.rectangle((0, H - FOOTER_H, W, H), fill=(12, 13, 18))
    blynk = {"off": ("Blynk: غير مفعّل", MUTED), "idle": ("Blynk: جاهز", MUTED),
             "ok": ("Blynk: متصل", GREEN), "error": ("Blynk: خطأ", RED)}[s.blynk_status]
    text(d, (14, H - FOOTER_H // 2), f"FPS {s.fps:.0f}", 13, MUTED, False, "lm")
    text(d, (90, H - FOOTER_H // 2), blynk[0], 13, blynk[1], False, "lm")
    text(d, (250, H - FOOTER_H // 2), "الصوت: " + ("مكتوم" if s.muted else "مفعّل"), 13, MUTED, False, "lm")
    text(d, (W - 14, H - FOOTER_H // 2), "Q خروج   R معايرة   M كتم   F ملء الشاشة   S لقطة", 13, MUTED, False, "rm")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
