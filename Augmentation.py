import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
)


def augment_and_save_images(directory, augmentations_per_image=5):
    for person_name in os.listdir(directory):
        person_path = os.path.join(directory, person_name)

        if os.path.isdir(person_path):
            print(f"Processing folder: {person_name}")

            for filename in os.listdir(person_path):
                file_path = os.path.join(person_path, filename)

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Skipping {file_path}: not a valid image.")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, 0)

                gen = datagen.flow(image, batch_size=1)
                for i in range(augmentations_per_image):
                    augmented_image = next(gen)[0].astype(np.uint8)

                    save_path = os.path.join(
                        person_path, f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg"
                    )
                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, augmented_image)
                    print(f"Saved augmented image: {save_path}")


dataset_directory = (
    "/Users/abdelwahab/3rd-year/deeplearing/Attendance-System/train_data/"
)
augment_and_save_images(dataset_directory)
