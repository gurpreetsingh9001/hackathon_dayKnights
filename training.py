import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
path = "../input/diabetic-retinopathy-resized/resized_train/resized_train"
data = "../input/diabetic-retinopathy-resized/resized_train/resized_train"
print('number of images in total - ',len(os.listdir(data)))
index = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels.csv") 
print('number of images in total - ',len(index))
#format correctly
index['image_name'] = [i+".jpeg" for i in index['image'].values]
index['level_binary'] = [i for i in index['level'].values]
index.head(10)
index.info()
train, val = train_test_split(index, test_size=0.2,random_state=42,stratify=index["level"])
train.shape, val.shape
train_datagen = ImageDataGenerator(rescale            = 1/255,
                                   shear_range        = 0.2,
                                   zoom_range         = 0.2,
                                   horizontal_flip    = True,
                                   rotation_range     = 40,
                                   width_shift_range  = 0.2,
                                   height_shift_range = 0.2)

val_datagen = ImageDataGenerator(rescale = 1/255)
train_set = train_datagen.flow_from_dataframe(
    train,
    "../input/diabetic-retinopathy-resized/resized_train/resized_train",
    x_col="image_name",
    y_col="level",
    class_mode="raw",
    color_mode="rgb",
    batch_size=32,
    target_size=(64, 64))

val_set = val_datagen.flow_from_dataframe(
    val,
    "../input/diabetic-retinopathy-resized/resized_train/resized_train",
    x_col="image_name",
    y_col="level",
    class_mode="raw",
   color_mode="rgb",
    batch_size=32,
    target_size=(64, 64))
    
model = Sequential()

model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation = 'softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_train = model.fit(
    train_set,
    #steps_per_epoch = 200,
    epochs = 5,
    validation_data = val_set,
    verbose=1
   # validation_steps = 100
)
model.save('diabetic retinopathy')