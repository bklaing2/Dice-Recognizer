import cv2
import tensorflow as tf


DATADIR = '../ml-data/dice-set/'
MODELDIR = '../gen/models/'
MODEL = 'd12-d4-conv-64_1-dense-64_16-batch_5-epoch.model'

CATEGORIES = ['d12', 'd4']
IMG_SIZE = 128

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model(MODELDIR + MODEL)

for i in range(20):
    prediction = model.predict([prepare(f'{DATADIR}/valid/{CATEGORIES[0]}/{i}.jpg')])
    prediction = CATEGORIES[round(prediction[0][0])]
    print(prediction)

print ('\n\n')

for i in range(20):
    prediction = model.predict([prepare(f'{DATADIR}/valid/{CATEGORIES[1]}/{i}.jpg')])
    prediction = CATEGORIES[round(prediction[0][0])]
    print(prediction)

print ('\n\n')

# TESTS = [
#     '../gen/die2.jpg',
#     '../gen/die3.jpg',
#     '../gen/die4.jpg',
# ]
#
# for test in TESTS:
#     prediction = model.predict([prepare(test)])
#     prediction = CATEGORIES[round(prediction[0][0])]
#     print(prediction)