import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

DATADIR = '../ml-data/dice-set/train/'
MODELDIR = '../gen/models/'

# Load data
X = pickle.load(open(DATADIR + 'X.pickle', 'rb'))
y = pickle.load(open(DATADIR + 'y.pickle', 'rb'))

# Normalizing data (255 is max, 0 is min)
X = np.array(X / 255.0)
y = np.array(y)


# Build model
# model = Sequential()
#
# model.add(Conv2D(8, (4,4), input_shape=X.shape[1:])) # Input layer (DON'T FULLY UNDERSTAND INPUT_SHAPE)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(8,8)))
#
# model.add(Conv2D(16, (2,2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(4,4)))
#
# # model.add(Conv2D(32, (3,3)))
# # model.add(Activation('relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
#
# # model.add(Conv2D(16, (3,3)))
# # model.add(Activation('relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Dropout(0.1)) # Dropout at 10%
#
# model.add(Flatten())
# # model.add(Dense(64))
# # model.add(Activation('relu'))
# #
# # model.add(Dense(32))
# # model.add(Activation('relu'))
#
# model.add(Dense(4)) # Output layer
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(X, y, batch_size=64, epochs=100, validation_split=0.3)



model = Sequential()

model.add(Conv2D(8, (5, 5), activation='relu', padding='same', input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(GlobalAveragePooling2D())

model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(X, y, batch_size=16, epochs=10, verbose=1, validation_split=0.1)


model.save(MODELDIR + 'test-guys.model')