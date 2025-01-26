import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

class DistillationModel(tf.keras.Model):
    def __init__(self, teacher_model, student_model, alpha, temperature):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha = alpha
        self.temperature = temperature

    def compile(self, optimizer, metrics):
        super(DistillationModel, self).compile()
        self.optimizer = optimizer
        self.train_metrics = metrics

    def train_step(self, data):
        x, y = data
        teacher_preds = self.teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student_model(x, training=True)
            ce_loss = tf.keras.losses.BinaryCrossentropy()(y, student_preds)
            kl_loss = tf.keras.losses.KLDivergence()(
                tf.nn.softmax(teacher_preds / self.temperature),
                tf.nn.softmax(student_preds / self.temperature)
            )
            loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        grads = tape.gradient(loss, self.student_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights))

        for metric in self.train_metrics:
            metric.update_state(y, student_preds)

        return {"loss": loss, **{m.name: m.result() for m in self.train_metrics}}

    def test_step(self, data):
        x, y = data
        student_preds = self.student_model(x, training=False)
        loss = tf.keras.losses.BinaryCrossentropy()(y, student_preds)

        for metric in self.train_metrics:
            metric.update_state(y, student_preds)

        return {"loss": loss, **{m.name: m.result() for m in self.train_metrics}}

    def call(self, inputs, training=False):
        return self.student_model(inputs, training=training)

# Example usage
teacher_model = load_model("teacher_model.h5")

# Student Model
student_model = Sequential([
    Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=(44, 116, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name="student")

# Ultra-Light Student Model
ultra_light_student_model = Sequential([
    Conv2D(4, kernel_size=3, activation='relu', padding='same', input_shape=(44, 116, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
], name="ultra_light_student")
