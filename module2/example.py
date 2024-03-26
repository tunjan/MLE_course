import os
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

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
    dataset = dataset.batch(200)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



def build_model():
    model = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape=(28, 28)),
             tf.keras.layers.Dense(200, activation='relu'),
             tf.keras.layers.Dense(10)
    ])



    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def load_and_preprocess_image(image_path):
    """Loads and preprocesses a single image for inference."""
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(28, 28)
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
model.save("output/mnist_model.keras")

# Evaluation on the test set
test_loss, test_accuracy = model.evaluate(ds_test_preprocessed)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

predictions = model.predict(ds_test_preprocessed) 
predicted_classes = np.argmax(predictions, axis=1)

true_labels = []
for x, y in ds_test_preprocessed:
    true_labels.extend(y.numpy())  # Extend for each element in the batch

# Create the DataFrame
df = pd.DataFrame({'Predicted Class': predicted_classes, 'True Label': true_labels})
df.to_csv('output/test_predictions.csv', index=False) 
