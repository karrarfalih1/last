"""ملف المريض (JSON): معاني أنماط الرمش وتوقيتاتها لكل مريض، دون تعديل الكود.

مثال: patients/default.json
    python main.py --profile patients/ahmed.json
"""
import json

import config

# مفتاح في JSON -> اسم الإعداد في config
TIMING_KEYS = {
    "deliberate_blink_min_seconds": "DELIBERATE_BLINK_MIN_SECONDS",
    "long_blink_seconds": "LONG_BLINK_SECONDS",
    "pattern_timeout": "PATTERN_TIMEOUT",
    "emergency_cooldown": "EMERGENCY_COOLDOWN",
    "scan_interval": "SCAN_INTERVAL",
}
CUSTOM_MENU = None     # قائمة مسح مخصصة من الملف (أو None = DEFAULT_MENU)


class ProfileError(ValueError):
    pass


def load_profile(path):
    """يقرأ الملف ويتحقق منه ويطبّقه على config. يُرجع اسم المريض."""
    with open(path, encoding="utf-8") as f:
        p = json.load(f)

    msgs = p.get("blink_messages")
    if msgs is not None:
        try:
            msgs = {int(k): str(v) for k, v in msgs.items()}
        except ValueError:
            raise ProfileError("مفاتيح blink_messages يجب أن تكون أرقاماً (عدد الرمشات)")
        if any(k < 2 for k in msgs):
            raise ProfileError("أقل نمط مسموح رمشتان (الرمشة الواحدة تحدث طبيعياً)")
        config.BLINK_MESSAGES = dict(sorted(msgs.items()))

    if "long_blink_message" in p:
        config.LONG_BLINK_MESSAGE = str(p["long_blink_message"])

    for key, attr in TIMING_KEYS.items():
        if key in p.get("timing", {}):
            value = float(p["timing"][key])
            if value <= 0:
                raise ProfileError(f"{key} يجب أن يكون موجباً")
            setattr(config, attr, value)
    if config.DELIBERATE_BLINK_MIN_SECONDS >= config.LONG_BLINK_SECONDS:
        raise ProfileError("مدة الرمشة المتعمّدة يجب أن تكون أقل من مدة رمشة الطوارئ")

    if "blink_signal" in p:
        if p["blink_signal"] not in ("ear", "blendshape", "fused"):
            raise ProfileError("blink_signal يجب أن يكون ear أو blendshape أو fused")
        config.BLINK_SIGNAL = p["blink_signal"]

    if "menu_pattern" in p:
        config.MENU_PATTERN = int(p["menu_pattern"]) if p["menu_pattern"] else 0   # 0 = بدون قائمة
    if config.MENU_PATTERN and (config.MENU_PATTERN < 2 or config.MENU_PATTERN in config.BLINK_MESSAGES):
        raise ProfileError("menu_pattern يجب أن يكون ≥2 ولا يتعارض مع أنماط blink_messages")

    global CUSTOM_MENU
    if "menu" in p:
        menu = p["menu"]
        if not menu or not all(c.get("title") and c.get("items") for c in menu):
            raise ProfileError("كل فئة في menu تحتاج title وقائمة items غير فارغة")
        CUSTOM_MENU = menu

    return p.get("name", "")
