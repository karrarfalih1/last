"""
تقييم كشف الرمش على تسجيلات record_protocol.py — بنفس خط المعالجة المستخدم في البرنامج (pipeline.py).

يقارن عدة إعدادات (دراسة استئصال / Ablation):
  ear | blendshape | fused | fused بدون فحص زاوية الرأس
ويحسب لكل إعداد: نسبة اكتشاف كل نمط، الإنذارات الكاذبة، والإنذارات الكاذبة في الساعة.

التشغيل:
  python evaluate_blink.py recordings/20261115_1420_volunteer1 [recordings/... ...]
  python evaluate_blink.py --all          # كل التسجيلات
الناتج: reports/blink_eval.md + reports/blink_eval_details.csv
"""
import argparse
import csv
import glob
import json
import os

import cv2

import config
from face_tracker import FaceTracker
from pipeline import BlinkPipeline

VARIANTS = [
    ("ear", True, "EAR فقط"),
    ("blendshape", True, "Blendshape فقط"),
    ("fused", True, "مدمج (الافتراضي)"),
    ("fused", False, "مدمج بدون فحص زاوية الرأس"),
]


def expected_of(msg):
    return "emergency" if msg.kind == "emergency" else msg.blinks


def score(messages, segments, grace=None):
    """messages: [(t, expected_key)]. يُرجع (hits لكل نوع, expected لكل نوع, false_alarms[(t, key, segment)]).
    الرسالة تُحسب إصابة إذا وقعت داخل مقطع يتوقعها (مع مهلة بعد نهايته لانتظار اكتمال النمط)."""
    grace = config.PATTERN_TIMEOUT + 1.0 if grace is None else grace
    hit_segments = set()
    hits, expected, false_alarms = {}, {}, []
    for i, s in enumerate(segments):
        if s["expected"] is not None:
            expected[s["expected"]] = expected.get(s["expected"], 0) + 1
    for t, key in messages:
        match = next((i for i, s in enumerate(segments)
                      if s["expected"] == key and i not in hit_segments and s["start"] <= t <= s["end"] + grace), None)
        if match is None:
            seg = next((s["instruction"] for s in segments if s["start"] <= t < s["end"]), "—")
            false_alarms.append((t, key, seg))
        else:
            hit_segments.add(match)
            hits[key] = hits.get(key, 0) + 1
    return hits, expected, false_alarms


def extract_observations(rec_dir, tracker_factory=FaceTracker):
    """يمرّر الفيديو على FaceLandmarker مرة واحدة ويُرجع [(t, FaceObs|None)]."""
    with open(os.path.join(rec_dir, "frames.json")) as f:
        times = json.load(f)
    cap = cv2.VideoCapture(os.path.join(rec_dir, "video.mp4"))
    tracker = tracker_factory()
    obs = []
    for t in times:
        ok, frame = cap.read()
        if not ok:
            break
        obs.append((t, tracker.process(frame, timestamp_ms=t * 1000)))
    cap.release()
    tracker.close()
    return obs


def run_variant(observations, signal, gate):
    pipe = BlinkPipeline(signal, pose_gate=gate)
    msgs = []
    for t, o in observations:
        m = pipe.step(o, t)
        if m:
            msgs.append((t, expected_of(m)))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dirs = sorted(glob.glob("recordings/*")) if args.all else args.recordings
    if not dirs:
        raise SystemExit("لا توجد تسجيلات. سجّل أولاً: python record_protocol.py --name volunteer1")

    totals = {v[2]: {"hits": {}, "expected": {}, "fa": 0, "hours": 0.0} for v in VARIANTS}
    details = []
    for d in dirs:
        with open(os.path.join(d, "protocol.json"), encoding="utf-8") as f:
            proto = json.load(f)
        print(f"… {d}")
        observations = extract_observations(d)
        face_rate = sum(o is not None for _, o in observations) / max(len(observations), 1)
        print(f"   إطارات={len(observations)}  نسبة ظهور الوجه={face_rate:.0%}")
        for signal, gate, label in VARIANTS:
            msgs = run_variant(observations, signal, gate)
            hits, expected, fas = score(msgs, proto["segments"])
            tot = totals[label]
            for k, v in expected.items():
                tot["expected"][k] = tot["expected"].get(k, 0) + v
            for k, v in hits.items():
                tot["hits"][k] = tot["hits"].get(k, 0) + v
            tot["fa"] += len(fas)
            tot["hours"] += proto["duration"] / 3600
            for t, key, seg in fas:
                details.append([d, label, f"{t:.1f}", key, seg])

    keys = sorted({k for v in totals.values() for k in v["expected"]}, key=lambda k: (k == "emergency", str(k)))
    name = lambda k: "طوارئ" if k == "emergency" else f"{k} رمشات"
    lines = ["# تقييم كشف الرمش", "",
             f"التسجيلات: {len(dirs)}  ·  المدة الكلية: {sum(v['hours'] for v in totals.values()) / len(VARIANTS) * 60:.1f} دقيقة", "",
             "| " + " | ".join(["الإعداد"] + [f"اكتشاف {name(k)}" for k in keys]
                               + ["الاكتشاف الكلي", "إنذارات كاذبة", "كاذبة/ساعة"]) + " |",
             "|" + "---|" * (len(keys) + 4)]
    for _, _, label in VARIANTS:
        t = totals[label]
        all_hits, all_exp = sum(t["hits"].values()), sum(t["expected"].values())
        cells = ([label] + [f"{t['hits'].get(k, 0)}/{t['expected'][k]}" for k in keys]
                 + [f"**{all_hits / all_exp:.0%}**" if all_exp else "—", str(t["fa"]),
                    f"{t['fa'] / max(t['hours'], 1e-9):.1f}"])
        lines.append("| " + " | ".join(cells) + " |")
    os.makedirs("reports", exist_ok=True)
    with open("reports/blink_eval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("reports/blink_eval_details.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["recording", "variant", "time", "detected", "during_instruction"])
        w.writerows(details)
    print("\n".join(lines))
    print("\nالتفاصيل (كل إنذار كاذب وأثناء أي تعليمة وقع): reports/blink_eval_details.csv")


if __name__ == "__main__":
    main()
