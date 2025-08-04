import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras import layers, models, callbacks, regularizers

# Load EMNIST ByClass dataset
emnist_data, emnist_info = tfds.load('emnist/byclass', with_info=True, as_supervised=True)
num_classes = emnist_info.features['label'].num_classes

train_ds_raw = emnist_data['train']
test_ds_raw = emnist_data['test']

BATCH_SIZE = 1024

# Fix orientation: rotate 270° CCW and flip left-right
def fix_orientation(image, label):
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    return image, label

# Normalize and add channel dimension
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

# Prepare datasets
train_ds = (
    train_ds_raw
    .map(fix_orientation, num_parallel_calls=tf.data.AUTOTUNE)
    .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds_raw
    .map(fix_orientation, num_parallel_calls=tf.data.AUTOTUNE)
    .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Augmentation layers
augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(factor=0.1),
    layers.RandomContrast(0.1),
])

# Residual block with L2 regularization
def residual_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Core ResNet model (deeper + more filters)
def build_core_resnet(input_shape=(28,28,1), num_classes=62):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1: 3x blocks with 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    # Stage 2: 3x blocks with 128 filters
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Stage 3: 3x blocks with 256 filters
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)

    return models.Model(inputs, outputs)

# Training wrapper (with augmentation)
def build_training_model(input_shape=(28,28,1), num_classes=62):
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs)
    core_model = build_core_resnet(input_shape, num_classes)
    outputs = core_model(x)
    return models.Model(inputs, outputs), core_model

# Create models
training_model, core_model = build_training_model()

# Compile training model
training_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_accuracy'
)

# Train model
history = training_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=[lr_scheduler, early_stopping, checkpoint]
)

# Evaluate final (clean) core model
core_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

test_loss, test_acc = core_model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Load class names from the EMNIST info
class_names = [str(i) for i in range(num_classes)]

# Fetch a batch of test images and labels
test_images, test_labels = next(iter(test_ds.unbatch().batch(64)))

# Choose random indices
num_to_plot = 9
indices = random.sample(range(test_images.shape[0]), num_to_plot)
images = tf.gather(test_images, indices)
labels = tf.gather(test_labels, indices)

# Predict using core_model
pred_probs = core_model.predict(images)
pred_labels = tf.argmax(pred_probs, axis=1)

# Plot
plt.figure(figsize=(10, 10))
for i in range(num_to_plot):
    plt.subplot(3, 3, i + 1)
    plt.imshow(tf.squeeze(images[i]), cmap='gray')
    plt.title(f"Pred: {pred_labels[i].numpy()}, True: {labels[i].numpy()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Export SavedModel
core_model.export("HR_tf")

# Convert to TensorFlow.js
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model HR_tf tfjs_model_HR_tf
