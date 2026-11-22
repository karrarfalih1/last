"""
تقييم ومقارنة نماذج المشاعر على نفس صور الاختبار (FER-2013 PrivateTest = 3,492 صورة صالحة).

لكل نموذج يحسب على تسميتين:
  - FER+   : الأغلبية من 10 مُرمِّزين (الأدق — المعيار الأساسي)
  - FER2013: التسمية الأصلية المشوّشة (للمقارنة مع الأرقام المنشورة)
ويُخرج: Accuracy, Macro-F1, Precision/Recall/F1 لكل فئة، مصفوفة الالتباس (صورة)، وزمن الاستدلال.

التشغيل:  python eval_emotion.py --models emotion_model.keras emotion_model_v2.keras
الناتج:   reports/emotion_eval.md  +  reports/confusion_<model>.png  +  reports/training_curves_v2.png
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from ferplus_data import load_all

LABELS = config.EMOTION_LABELS


def per_class(y_true, y_pred, k=len(LABELS)):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return cm, precision, recall, f1


def predict(model_path, x):
    """تُرجع (الاحتمالات، زمن الاستدلال لصورة واحدة بالميلي ثانية)."""
    imgs = x[..., None].astype("float32")
    if model_path.endswith(".tflite"):
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        probs = []
        t0 = time.perf_counter()
        for im in imgs:
            it.set_tensor(inp["index"], im[None])
            it.invoke()
            probs.append(it.get_tensor(out["index"])[0])
        ms = (time.perf_counter() - t0) / len(imgs) * 1000
        return np.array(probs), ms
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    probs = model.predict(imgs, batch_size=128, verbose=0)
    one = imgs[:1]
    model.predict(one, verbose=0)
    t0 = time.perf_counter()
    for _ in range(30):
        model.predict(one, verbose=0)                 # نفس طريقة الاستدعاء في البرنامج (صورة واحدة)
    ms = (time.perf_counter() - t0) / 30 * 1000
    return probs, ms


def plot_confusion(cm, title, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (FER+ majority)"); ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(path_json, out):
    with open(path_json) as f:
        h = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(h["accuracy"], label="train"); axes[0].plot(h["val_accuracy"], label="val")
    axes[1].plot(h["loss"], label="train"); axes[1].plot(h["val_loss"], label="val")
    for ax, t in zip(axes, ("Accuracy", "Loss")):
        ax.axvline(2.5, color="gray", ls="--", lw=0.8)
        ax.set_title(t); ax.set_xlabel("epoch"); ax.legend()
    axes[0].text(2.7, min(h["val_accuracy"]), "fine-tuning →", color="gray", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["emotion_model.keras", "emotion_model_v2.keras"])
    args = ap.parse_args()
    x, y_soft, y_old = load_all()["test"]
    y_plus = y_soft.argmax(1)
    os.makedirs("reports", exist_ok=True)

    rows, detail = [], []
    for path in args.models:
        if not os.path.exists(path):
            print(f"تخطّي {path} (غير موجود)")
            continue
        probs, ms = predict(path, x)
        pred = probs.argmax(1)
        name = os.path.splitext(os.path.basename(path))[0]
        cm, p, r, f1 = per_class(y_plus, pred)
        _, _, _, f1_old = per_class(y_old, pred)
        size_mb = os.path.getsize(path) / 1e6
        rows.append(f"| `{os.path.basename(path)}` | **{np.mean(pred == y_plus):.1%}** | {f1.mean():.3f} | "
                    f"{np.mean(pred == y_old):.1%} | {f1_old.mean():.3f} | {ms:.1f} ms | {size_mb:.1f} MB |")
        detail.append(f"\n### `{os.path.basename(path)}` — لكل فئة (FER+)\n\n| الفئة | Precision | Recall | F1 | عدد الصور |\n|---|---|---|---|---|")
        for i, lab in enumerate(LABELS):
            detail.append(f"| {config.EMOTION_LABELS_AR[lab]} ({lab}) | {p[i]:.2f} | {r[i]:.2f} | {f1[i]:.2f} | {cm[i].sum()} |")
        plot_confusion(cm, f"{name}  (acc {np.mean(pred == y_plus):.1%})", f"reports/confusion_{name}.png")
        detail.append(f"\n![confusion](confusion_{name}.png)")
        print(f"{path}: FER+ acc={np.mean(pred == y_plus):.2%}  macroF1={f1.mean():.3f}  "
              f"FER2013 acc={np.mean(pred == y_old):.2%}  {ms:.1f} ms/face")

    if os.path.exists("reports/v2_history.json"):
        plot_history("reports/v2_history.json", "reports/training_curves_v2.png")

    agree = np.mean(y_plus == y_old)
    md = ["# تقييم نماذج المشاعر", "",
          f"صور الاختبار: {len(x)} (FER-2013 PrivateTest بعد حذف غير الصالح وفق FER+). "
          f"اتفاق التسمية الأصلية مع أغلبية FER+: **{agree:.1%}** فقط — أي أن ثلث تسميات FER-2013 مشكوك فيها.", "",
          "| النموذج | دقة (FER+) | Macro-F1 (FER+) | دقة (FER-2013 الأصلية) | Macro-F1 (الأصلية) | زمن/وجه | الحجم |",
          "|---|---|---|---|---|---|---|", *rows, *detail]
    if os.path.exists("reports/training_curves_v2.png"):
        md += ["", "### منحنيات تدريب v2", "", "![curves](training_curves_v2.png)"]
    with open("reports/emotion_eval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("التقرير: reports/emotion_eval.md")


if __name__ == "__main__":
    main()
