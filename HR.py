import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

df_test = pd.read_csv("./emnist-letters-test.csv", header=None)
df_train = pd.read_csv("./emnist-letters-train.csv", header=None)

def prep(X_flat):
    X_reshaped = X_flat.reshape(-1, 28, 28)
    X_rotated = np.rot90(X_reshaped, k=1, axes=(1, 2))
    X_flipped = np.fliplr(X_rotated)
    X_fixed = X_flipped[..., np.newaxis]
    return X_fixed.astype(np.float32) / 255.0

X_test = prep(df_test.drop(columns=[0]).values)
y_test = df_test[0].values - 1

X_train = prep(df_train.drop(columns=[0]).values)
y_train = df_train[0].values - 1

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

y_train_split = to_categorical(y_train_split, 26)
y_val_split   = to_categorical(y_val_split, 26)
y_test        = to_categorical(y_test, 26)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(26),
    y=np.argmax(y_train_split, axis=1)
)
class_weights = dict(enumerate(class_weights))

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15
)
val_datagen = ImageDataGenerator()

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

inputs = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, 64)
x = residual_block(x, 64)

x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)

x = residual_block(x, 256, stride=2)   
x = residual_block(x, 256)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(26, activation='softmax')(x)

model = models.Model(inputs, outputs)

cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=len(X_train_split) // 128 * 10,
    t_mul=2.0,
    m_mul=0.8
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(cosine_decay),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', monitor='val_loss', save_best_only=True, verbose=1
)

model.fit(
    train_datagen.flow(X_train_split, y_train_split, batch_size=128),
    validation_data=val_datagen.flow(X_val_split, y_val_split, batch_size=128),
    epochs=300,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

model.load_weights('best_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f"Test accuracy: {test_acc:.4f}")

model.save('HR', save_format='tf')

