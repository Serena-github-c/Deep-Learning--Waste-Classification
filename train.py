import numpy as np
from PIL import Image, ImageFile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from download_data import download_dataset

# Ensure the dataset is available before training
download_dataset()


# Paths for datasets
train_path = "./Waste Segregation Image Dataset/train"
val_path = "./Waste Segregation Image Dataset/val"
test_path = "./Waste Segregation Image Dataset/test"

# Categories
categories = ["ewaste", 
              "food_waste", 
              "leaf_waste", 
              "metal_cans", 
              "paper_waste", 
              "plastic_bags", 
              "plastic_bottles", 
              "wood_waste"]


ImageFile.LOAD_TRUNCATED_IMAGES = True
input_size = 299
learning_rate = 0.001
size_inner=100 


## Data
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    train_path,
    target_size=(299, 299),
    batch_size=32,
    classes=categories
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)


val_ds = val_gen.flow_from_directory(
    val_path,
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False,
    classes=categories
)


## Model
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']    


def make_model(input_size=299, learning_rate=0.01, size_inner=100):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = GlobalAveragePooling2D()(base)
    
    inner = Dense(size_inner, activation='relu')(vectors)    
    outputs = Dense(8)(inner)
    
    model = Model(inputs, outputs)
    
    #########################################

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model




checkpoint_299 = keras.callbacks.ModelCheckpoint(
    'xception_299_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
)

history = model.fit(train_ds, epochs=30, validation_data=val_ds,
                   callbacks=[checkpoint_299])


