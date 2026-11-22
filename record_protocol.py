"""
تسجيل جلسة تقييم موحّدة لكشف الرمش (لمتطوع واحد، ~4 دقائق).

يعرض على الشاشة تعليمات مجدولة (ارمش مرتين، انظر للأسفل، تكلّم…) ويسجّل:
  recordings/<وقت>/video.mp4      — الفيديو
  recordings/<وقت>/frames.json    — توقيت كل إطار بالثواني
  recordings/<وقت>/protocol.json  — المقاطع: البداية والنهاية والمتوقع (عدد رمشات / طوارئ / لا شيء)
ثم:  python evaluate_blink.py recordings/<وقت>

التشغيل:  python record_protocol.py [--source 0] [--name volunteer1]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ui import AMBER, GREEN, MUTED, RED, TEXT, text

# (التعليمة، المدة بالثواني، المتوقع) — المتوقع: عدد الرمشات، "emergency"، أو None (يجب ألا يُرسل شيء)
REST = ("استرح وانظر للكاميرا", 4, None)
PROTOCOL = [
    ("معايرة: انظر للكاميرا بوجه هادئ", 6, None),
    ("اجلس طبيعياً وارمش كالمعتاد", 20, None),
    ("ارمش مرتين ببطء", 7, 2), REST,
    ("ارمش 3 مرات ببطء", 8, 3), REST,
    ("أغلق عينيك ثانيتين ثم افتحهما", 6, "emergency"), REST,
    ("انظر للأسفل (كأنك تقرأ)", 5, None), REST,
    ("ارمش مرتين ببطء", 7, 2), REST,
    ("تكلّم بصوت عالٍ: عدّ من 1 إلى 15", 10, None), REST,
    ("ارمش 4 مرات ببطء", 9, 4), REST,
    ("التفت يساراً ثم يميناً ببطء", 6, None), REST,
    ("ارمش 3 مرات ببطء", 8, 3), REST,
    ("ابتسم ثم اضحك", 5, None), REST,
    ("أغلق عينيك ثانيتين ثم افتحهما", 6, "emergency"), REST,
    ("ارمش مرتين ببطء", 7, 2), REST,
    ("ارمش 4 مرات ببطء", 9, 4), REST,
    ("ارمش 3 مرات ببطء", 8, 3), REST,
    ("اجلس طبيعياً وارمش كالمعتاد", 20, None),
]


def build_segments(protocol=PROTOCOL):
    t, segs = 0.0, []
    for instr, dur, expected in protocol:
        segs.append({"start": t, "end": t + dur, "instruction": instr, "expected": expected})
        t += dur
    return segs


def overlay(frame, seg, remaining, progress, name):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    exp = seg["expected"]
    col = RED if exp == "emergency" else GREEN if exp else MUTED
    d.rectangle((0, 0, w, 96), fill=(0, 0, 0, 170))
    text(d, (w // 2, 36), seg["instruction"], 26, col if exp else TEXT, True, "mm")
    text(d, (w // 2, 74), f"{remaining:.0f}", 22, AMBER, True, "mm")
    d.rectangle((0, h - 8, w * progress, h), fill=AMBER)
    text(d, (10, h - 16), f"REC  {name}", 13, RED, True, "ld")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--name", default="volunteer")
    ap.add_argument("--duration", type=float, help="إيقاف مبكر (للاختبار فقط)")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    segs = build_segments()
    total = segs[-1]["end"] if args.duration is None else min(args.duration, segs[-1]["end"])
    out_dir = os.path.join("recordings", time.strftime("%Y%m%d_%H%M%S") + f"_{args.name}")
    os.makedirs(out_dir)

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("تعذّر فتح الكاميرا")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(os.path.join(out_dir, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    times = []
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            t = time.time() - t0
            if t >= total:
                break
            writer.write(frame)                       # الفيديو بدون التعليمات
            times.append(round(t, 4))
            seg = next(s for s in segs if s["start"] <= t < s["end"])
            cv2.imshow("Blink evaluation protocol  (Q = abort)", overlay(frame, seg, seg["end"] - t, t / total, args.name))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("أُلغي التسجيل")
                break
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump(times, f)
    with open(os.path.join(out_dir, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "duration": times[-1] if times else 0,
                   "segments": [s for s in segs if s["start"] < (times[-1] if times else 0)]},
                  f, ensure_ascii=False, indent=1)
    print(f"تم الحفظ في {out_dir}  ({len(times)} إطار، {len(times) / max(times[-1], 1e-6):.1f} FPS)")
    print(f"للتقييم:  python evaluate_blink.py {out_dir}")


if __name__ == "__main__":
    main()
