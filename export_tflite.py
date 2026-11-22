"""تحويل نموذج المشاعر إلى TFLite (أسرع على المعالج، وأصغر، ويعمل على Raspberry Pi).

    python export_tflite.py emotion_model_v2.keras            -> emotion_model_v2.tflite (float16)
بعدها في config.py:  EMOTION_MODEL_PATH = "emotion_model_v2.tflite"
"""
import sys

import tensorflow as tf
from tensorflow import keras

src = sys.argv[1] if len(sys.argv) > 1 else "emotion_model_v2.keras"
dst = src.rsplit(".", 1)[0] + ".tflite"

model = keras.models.load_model(src)
# طبقات الزيادة (augmentation) تعمل فقط أثناء التدريب؛ نصدّر مسار الاستدلال بدالة ثابتة الإدخال
fn = tf.function(lambda x: model(x, training=False),
                 input_signature=[tf.TensorSpec([1, 48, 48, 1], tf.float32)])
conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
with open(dst, "wb") as f:
    f.write(conv.convert())
print(f"تم: {dst}")
