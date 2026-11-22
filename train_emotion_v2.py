"""
الإصدار 2 من نموذج المشاعر: Transfer Learning بـ MobileNetV2 (أوزان ImageNet) على FER+ .

التحسينات عن الإصدار 1 (train_emotion.py):
  - تسميات FER+ الناعمة بدل تسميات FER-2013 المشوّشة.
  - شبكة MobileNetV2 مُدرّبة مسبقاً + Fine-tuning على مرحلتين.
  - أوزان للعيّنات حسب ندرة الفئة (الخوف والانزعاج نادران).
  - نفس واجهة الإدخال القديمة (صورة رمادية 48×48 بقيم 0–255)، فلا يتغير emotion_detector.py.

التشغيل:  python train_emotion_v2.py      الناتج: emotion_model_v2.keras + reports/v2_history.json
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from ferplus_data import load_all

OUT = "emotion_model_v2.keras"
SIZE = 96
BATCH = 64
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 30

tf.keras.utils.set_random_seed(42)
data = load_all()
(x_tr, y_tr, _), (x_va, y_va, _) = data["train"], data["val"]

# أوزان العيّنات: الجذر التربيعي لعكس التكرار (تخفيف الاختلال بدون مبالغة)
counts = np.bincount(y_tr.argmax(1), minlength=len(config.EMOTION_LABELS))
class_w = np.sqrt(counts.sum() / (len(counts) * counts))
sample_w = class_w[y_tr.argmax(1)].astype("float32")
print("أوزان الفئات:", dict(zip(config.EMOTION_LABELS, np.round(class_w, 2))))


def ds(x, y, w=None, shuffle=False):
    x = x[..., None].astype("float32")
    d = tf.data.Dataset.from_tensor_slices((x, y) if w is None else (x, y, w))
    if shuffle:
        d = d.shuffle(len(x), seed=42)
    return d.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds = ds(x_tr, y_tr, sample_w, shuffle=True), ds(x_va, y_va)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
], name="augment")

base = keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet")

inputs = keras.Input(shape=(48, 48, 1), name="gray_48x48_0_255")
x = augment(inputs)
x = layers.Resizing(SIZE, SIZE)(x)
x = layers.Concatenate()([x, x, x])                 # رمادي -> 3 قنوات
x = layers.Rescaling(1 / 127.5, offset=-1)(x)       # مدخلات MobileNetV2 في [-1, 1]
x = base(x, training=False)                         # BatchNorm يبقى بوضع الاستدلال أثناء Fine-tuning
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(len(config.EMOTION_LABELS), activation="softmax")(x)
model = keras.Model(inputs, outputs, name="emotion_v2_mobilenetv2")

history = {}


def run(epochs, lr, tag, callbacks=()):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=list(callbacks), verbose=2)
    for k, v in h.history.items():
        history.setdefault(k, []).extend(float(a) for a in v)
    print(f"[{tag}] best val_accuracy = {max(h.history['val_accuracy']):.4f}", flush=True)


# المرحلة 1: تدريب الرأس فقط
base.trainable = False
run(HEAD_EPOCHS, 1e-3, "head")

# المرحلة 2: Fine-tuning للشبكة كاملة بمعدل تعلّم صغير
base.trainable = True
run(FINETUNE_EPOCHS, 1e-4, "finetune", callbacks=[
    keras.callbacks.ModelCheckpoint(OUT, monitor="val_accuracy", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
])

os.makedirs("reports", exist_ok=True)
with open("reports/v2_history.json", "w") as f:
    json.dump(history, f)
print(f"تم. النموذج: {OUT} — شغّل eval_emotion.py للمقارنة.", flush=True)
