# USING THIS
# https://www.kaggle.com/mirlei/image-classification-of-dice-with-neural-network




import random
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


DATADIR = '../ml-data/dice/train'
MODELDIR = '../gen/models/'
CATEGORIES = ['d4', 'd6', 'd8', 'd10', 'd12', 'd20']


train_path = ('../ml-data/dice/train')
valid_path = ('../ml-data/dice/valid')

batch_size_train = 10
batch_size_valid = 10
targetsize = 48

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(targetsize, targetsize), classes=['d4', 'd6', 'd8', 'd10', 'd12', 'd20'], batch_size=batch_size_train)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(targetsize, targetsize), classes=['d4', 'd6', 'd8', 'd10', 'd12', 'd20'], batch_size=batch_size_valid)

train_num = len(train_batches)
val_num = len(valid_batches)


def plots(ims, figsize=(20,10), rows=1, interp= False, titles= None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)

        if ims.shape[-1] != 3:
            ims = ims.transpose((0,1,2,3))

    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) %2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')

        if titles is not None:
            sp.set_title(titles[i], fontsize=12)

        cv2.imshow('im', ims[i])#, interpolation=None if interp else 'none')
        cv2.waitKey(0)


# imgs, labels = next(train_batches)
# plots(imgs, titles=labels)

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(targetsize, targetsize, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax'),

])


model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics= ['accuracy'])
history = model.fit_generator(train_batches, steps_per_epoch= train_num , validation_data=valid_batches, validation_steps= val_num, epochs=15, verbose=2)

# model.fit(X, y, batch_size=16, epochs=10, verbose=1, validation_split=0.1)

model.save(MODELDIR + 'test-guys.model')