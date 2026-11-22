"""
تدريب CNN لتمييز المشاعر على FER-2013.

1) حمّل البيانات من Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2) فك الضغط بحيث يصبح الشكل:
     data/train/angry/*.jpg ... data/train/surprise/*.jpg
     data/test/angry/*.jpg  ... data/test/surprise/*.jpg
3) python train_emotion.py
الناتج: emotion_model.keras  (الدقة المتوقعة تقريباً 60–66% وهي طبيعية لهذا الـ dataset)
"""
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import config

IMG = (48, 48)
BATCH = 64
EPOCHS = 40

train_ds = keras.utils.image_dataset_from_directory(
    "data/train", labels="inferred", label_mode="categorical", color_mode="grayscale",
    image_size=IMG, batch_size=BATCH, class_names=config.EMOTION_LABELS, shuffle=True, seed=42)
test_ds = keras.utils.image_dataset_from_directory(
    "data/test", labels="inferred", label_mode="categorical", color_mode="grayscale",
    image_size=IMG, batch_size=BATCH, class_names=config.EMOTION_LABELS, shuffle=False)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.25)(x)


inputs = keras.Input(shape=(48, 48, 1))
x = augment(inputs)
x = layers.Rescaling(1.0 / 255)(x)
for f in (32, 64, 128, 256):
    x = conv_block(x, f)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(config.EMOTION_LABELS), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    train_ds, validation_data=test_ds, epochs=EPOCHS,
    callbacks=[
        keras.callbacks.ModelCheckpoint(config.EMOTION_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ])

loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.2%}")

# رسم منحنيات التدريب (مفيد لتقرير المشروع)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"])
plt.title("Accuracy"); plt.legend(["train", "val"])
plt.subplot(1, 2, 2); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"])
plt.title("Loss"); plt.legend(["train", "val"])
plt.tight_layout(); plt.savefig("training_curves.png")
print("تم حفظ النموذج و training_curves.png")
