import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings
import time

import tensorflow as tf
from tensorflow import keras
from keras import layers, metrics
from keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt

start = time.time()
# Set the seed value for experiment reproducibility.
seed = 1842
tf.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')

#define image dataset
# Data Augmentation
image_generator = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2],# brightness
        validation_split=0.2,)

#Train & Validation Split
train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='./images',
                                                 shuffle=True,
                                                 target_size=(188, 188),
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='./images',
                                                 shuffle=True,
                                                 target_size=(188, 188),
                                                 subset="validation",
                                                 class_mode='categorical')


model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = [188, 188,3]),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(129600, activation="relu"),
    keras.layers.Dense(64800, activation="relu"),
    keras.layers.Dense(1, activation="softmax")
])

model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['accuracy', 'AUC'])

print(model.summary())


history = model.fit(train_dataset, epochs=500, validation_data=validation_dataset)

pl_loss = history.history['loss']
pl_val_loss = history.history['val_loss']

epoch_count = range(1, len(pl_loss) + 1)

plt.plot(epoch_count, pl_loss, 'r-')
plt.plot(epoch_count, pl_val_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("loss_epoch.png")


loss, accuracy, auc = model.evaluate(validation_dataset)

with open('results.txt', 'a+') as file:
    file.write(f"EVALUATE RESULTS\n")
    file.write(f"Loss: {loss}\n")
    file.write(f"ACC: {accuracy}\n")
    file.write(f"AUC: {auc}\n")

model.save('cnn-model_try2')

end = time.time()
print(end - start)

with open('results.txt', 'a+') as file:
    file.write(f"Total Training Time\n")
    file.write(f"Time elapsed: {end-start}\n")
