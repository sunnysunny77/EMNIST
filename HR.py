import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler

NUM_CLASSES = 62
BATCH_SIZE = 64

df_test = pd.read_csv("./emnist-byclass-test.csv", header=None)
df_train = pd.read_csv("./emnist-byclass-train.csv", header=None)

scaler = MinMaxScaler()
X_train_flat = df_train.drop(columns=[0]).values 
X_test_flat = df_test.drop(columns=[0]).values 

X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

def prep(X_scaled):
    return np.fliplr(np.rot90(X_scaled.reshape(-1, 28, 28), 1, axes=(1, 2)))[..., None].astype(np.float32)

X_train = prep(X_train_scaled)
y_train_int = df_train[0].values
y_train = to_categorical(y_train_int, NUM_CLASSES)

X_test = prep(X_test_scaled)
y_test = to_categorical(df_test[0].values, NUM_CLASSES)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

class_weights = {i: w for i, w in enumerate(compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train_int))}

train_datagen = ImageDataGenerator(
     rotation_range=8,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

model = models.Sequential([
    
    layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),

    layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.35),

    layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation='softmax')
    
])

steps_per_epoch = -(-len(X_train_split) // BATCH_SIZE)

cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=steps_per_epoch * 10,
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
    patience=6,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', monitor='val_loss', save_best_only=True, verbose=1
)

model.fit(
    train_datagen.flow(X_train_split, y_train_split, batch_size=BATCH_SIZE),
    validation_data=(X_val_split, y_val_split),
    epochs=17,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

model.load_weights('best_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"test_acc: {test_acc} | test_loss: {test_loss}")

model.save('HR', save_format='tf')
#test_acc: 0.890614926815033 | test_loss: 0.9542901515960693