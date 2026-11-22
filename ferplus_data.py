"""تحميل FER-2013 مع تسميات FER+ المصحّحة (Microsoft) لاستخدامها في التدريب والتقييم.

المصادر:
  data_ferplus/fer2013.csv     — الصور (عمود pixels) من Kaggle: deadskull7/fer2013
  data_ferplus/fer2013new.csv  — أصوات 10 مُرمِّزين لكل صورة: github.com/microsoft/FERPlus
الصفوف في الملفين بنفس الترتيب.

التسمية الناتجة "ناعمة" (soft): توزيع أصوات المُرمِّزين على الفئات السبع، بدل فئة واحدة قد تكون خاطئة.
"""
import csv
import os

import numpy as np

import config

DIR = "data_ferplus"
# أعمدة FER+ المقابلة لفئاتنا بنفس ترتيب config.EMOTION_LABELS
FERPLUS_COLUMN = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happiness",
                  "neutral": "neutral", "sad": "sadness", "surprise": "surprise"}
SPLITS = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}


def _cache_path():
    return os.path.join(DIR, "ferplus_cache.npz")


def load_all():
    """تُرجع dict: split -> (images uint8 [N,48,48], soft_labels float32 [N,7], fer2013_label int [N])."""
    if os.path.exists(_cache_path()):
        z = np.load(_cache_path())
        return {s: (z[f"{s}_x"], z[f"{s}_y"], z[f"{s}_old"]) for s in SPLITS}

    with open(os.path.join(DIR, "fer2013.csv"), newline="") as f:
        base = list(csv.DictReader(f))
    with open(os.path.join(DIR, "fer2013new.csv"), newline="") as f:
        plus = list(csv.DictReader(f))
    assert len(base) == len(plus), "عدد الصفوف في الملفين غير متطابق"

    # ترتيب فئات FER-2013 الأصلية: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
    old_order = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    old_to_ours = [config.EMOTION_LABELS.index(n) for n in old_order]

    out = {s: ([], [], []) for s in SPLITS}
    usage_to_split = {v: k for k, v in SPLITS.items()}
    dropped = 0
    for b, p in zip(base, plus):
        assert b["Usage"] == p["Usage"], "ترتيب الصفوف غير متطابق"
        votes = np.array([float(p[FERPLUS_COLUMN[n]]) for n in config.EMOTION_LABELS])
        other = max(float(p["contempt"]), float(p["unknown"]), float(p["NF"]))
        votes[votes <= 1] = 0            # حذف الصوت المنفرد (مُرمِّز شاذ) كما في ورقة FER+
        if votes.sum() == 0 or other >= votes.max():
            dropped += 1                 # ليست وجهاً، أو غير واضحة، أو "ازدراء" (ليست ضمن فئاتنا)
            continue
        img = np.array(b["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        xs, ys, olds = out[usage_to_split[b["Usage"]]]
        xs.append(img)
        ys.append(votes / votes.sum())
        olds.append(old_to_ours[int(b["emotion"])])

    data = {s: (np.stack(x), np.array(y, np.float32), np.array(o, np.int64)) for s, (x, y, o) in out.items()}
    np.savez_compressed(_cache_path(), **{f"{s}_{k}": v for s, t in data.items() for k, v in zip(("x", "y", "old"), t)})
    print(f"[FER+] حُذفت {dropped} صورة غير صالحة. "
          + ", ".join(f"{s}={len(t[0])}" for s, t in data.items()))
    return data
