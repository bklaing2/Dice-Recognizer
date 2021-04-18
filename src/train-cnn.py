import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D

DATADIR = '../ml-data/dice-set/train/'
MODELDIR = '../gen/models/'

# Load data
X = pickle.load(open(DATADIR + 'X.pickle', 'rb'))
y = pickle.load(open(DATADIR + 'y.pickle', 'rb'))

# Normalizing data (255 is max, 0 is min)
X = np.array(X / 255.0)
y = np.array(y)


# Build model
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:])) # Input layer (DON'T FULLY UNDERSTAND INPUT_SHAPE)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1)) # Output layer
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, y, batch_size=16, epochs=5, validation_split=0.3)

model.save(MODELDIR + 'd12-d4-conv-64_1-dense-64_16-batch_5-epoch.model')