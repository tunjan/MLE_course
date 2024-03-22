import os
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_mnist_data():
    """Loads and splits the MNIST dataset."""
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_test, ds_info


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def preprocess_dataset(dataset):
    """Preprocesses the dataset with normalization, caching, shuffling, and batching."""
    dataset = dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(ds_info.splits["train"].num_examples)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model():
    """Defines the CNN model for MNIST classification."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def load_and_preprocess_image(image_path):
    """Loads and preprocesses a single image for inference."""
    # Load image (using your preferred library like PIL, OpenCV etc.)
    image = tf.keras.preprocessing.image.load_img(
        image_path, grayscale=True, target_size=(28, 28)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension
    return normalize_img(image_array, None)[0]  # Normalize


# Model Training
ds_train, ds_test, ds_info = load_mnist_data()
ds_train_preprocessed = preprocess_dataset(ds_train)
ds_test_preprocessed = preprocess_dataset(ds_test)

model = build_model()
model.fit(
    ds_train_preprocessed,
    epochs=6,
    validation_data=ds_test_preprocessed,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
)  # Add early stopping
model.save("mnist_model.keras")

# Model Inference
model = tf.keras.models.load_model("mnist_model.keras")

# Evaluation on the test set
test_loss, test_accuracy = model.evaluate(ds_test_preprocessed)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Prediction on a sample image
sample_image = load_and_preprocess_image("image.png")
prediction = model.predict(sample_image)
predicted_class = np.argmax(prediction[0])
print("Predicted class:", predicted_class)
