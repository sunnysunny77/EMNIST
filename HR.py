import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

df_test = pd.read_csv("./emnist-balanced-test.csv", header=None)
df_train = pd.read_csv("./emnist-balanced-train.csv", header=None)

def prep(X_flat):

    X_reshaped = X_flat.reshape(-1, 28, 28)
    
    X_rotated = np.rot90(X_reshaped, k=1, axes=(1, 2))
    X_flipped = np.fliplr(X_rotated)
    
    X_fixed = X_flipped[..., np.newaxis]
    
    return X_fixed.astype(np.float32) / 255.0

X_test = prep(df_test.drop(columns=[0]).values)
y_test = df_test[0].values

X_train = prep(df_train.drop(columns=[0]).values)
y_train = df_train[0].values

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
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

inputs = tf.keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, 32)
x = residual_block(x, 32)

x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)

x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(47, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
)

model.fit(
    train_datagen.flow(X_train_split, y_train_split, batch_size=32),
    validation_data=val_datagen.flow(X_val_split, y_val_split, batch_size=32),
    epochs=300,
    callbacks=[early_stop],
    verbose=2
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")

mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
    45: 'r', 46: 't',
}

predicted_labels = np.argmax(model.predict(X_test), axis=1)
predicted_chars = [mapping[label] for label in predicted_labels]
actual_chars = [mapping[label] for label in y_test]

print("First 10 predictions vs actual:")
for i in range(10):
    print(f"Predicted: {predicted_chars[i]} \t Actual: {actual_chars[i]}")

accuracy = np.mean(predicted_labels == y_test)
print(f"\nAccuracy: {accuracy:.4f}")
