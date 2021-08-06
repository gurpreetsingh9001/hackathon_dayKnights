import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random as an
from glob import glob
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split

path = "../input/diabetic-retinopathy-resized/resized_train/resized_train/"
data = "../input/diabetic-retinopathy-resized/resized_train/resized_train"
print('number of images in total - ',len(os.listdir(data)))
index = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels.csv") 
print('number of dataframes in total - ',len(index))
index.head()
#format correctly
index['image_name'] = [i+".jpeg" for i in index['image'].values]
index['level_binary'] = [i for i in index['level'].values]
    
index.head(10)
train, val = train_test_split(index, test_size=0.001,random_state=5,stratify=index["level"])
val.shape

val_datagen = ImageDataGenerator(rescale = 1/255)
val_set = val_datagen.flow_from_dataframe(
    val,
    "../input/diabetic-retinopathy-resized/resized_train/resized_train",
    x_col="image_name",
    y_col="level",
    class_mode="raw",
   color_mode="rgb",
    batch_size=32,
    target_size=(64, 64))
    
model = load_model("../input/disease-classifier/disease_classifier.h5")
 
# load an image and predict the class
def run_example():
    result = model.predict(val_set)
    for i in range(len(result)):
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(15, 5)
        legend = ['severe', 'mild', 'moderate', 'symptoms', 'negative']
        ax[0].bar(legend, result[i])
        ax[0].set_yticks(np.arange(0,1.1,0.1))
        ax[1].imshow(plt.imread(path+val.iloc[i].image_name))
        plt.show()

run_example()