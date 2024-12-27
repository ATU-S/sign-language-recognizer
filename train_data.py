import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir="sign_data", img_size=(128, 128)):
    """
    Loads image data and corresponding labels from the specified directory.

    Parameters:
        data_dir (str): Directory containing subdirectories for each sign.
        img_size (tuple): Size to resize the images. Default is (128, 128).

    Returns:
        tuple: (images, labels) where images is a NumPy array of shape (n, img_size[0], img_size[1], 3)
               and labels is a NumPy array of integer labels.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Ensure classes are sorted

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = tf.keras.utils.load_img(img_path, target_size=img_size)
                img = tf.keras.utils.img_to_array(img)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    images = np.array(images, dtype="float32") / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels, class_names

def build_model(input_shape, num_classes):
    """
    Builds and returns an enhanced CNN model for image classification.

    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for classification.

    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second convolutional layer
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Third convolutional layer
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Flatten the output and add dense layers
        Flatten(),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model(data_dir="sign_data", img_size=(128, 128), epochs=20, batch_size=32):
    """
    Trains an enhanced CNN model using the images and labels from the specified directory.

    Parameters:
        data_dir (str): Directory containing the image data.
        img_size (tuple): Size to resize the images. Default is (128, 128).
        epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Training batch size. Default is 32.

    Returns:
        tf.keras.Model: The trained model.
        list: Class names corresponding to the labels.
    """
    # Load data
    images, labels, class_names = load_data(data_dir, img_size)

    # Save class names to a file
    with open("class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(class_name + "\n")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Build model
    model = build_model(input_shape=img_size + (3,), num_classes=len(class_names))

    # Train model with data augmentation
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              epochs=epochs,
              validation_data=(X_test, y_test))

    # Save the model
    model.save("sign_language_model.h5")
    print("Model training complete and saved as 'sign_language_model.h5'.")
    print("Class names saved to 'class_names.txt'.")

    return model, class_names

if __name__ == "__main__":
    train_model()
