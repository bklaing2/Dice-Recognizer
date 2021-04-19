import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATADIR = '../ml-data/dice-set/'
MODELDIR = '../gen/models/'
MODEL = 'test-guys.model'

CATEGORIES = ['d4', 'd6', 'd8', 'd10', 'd12', 'd20']
IMG_SIZE = 48

def prepare(filepath):
    img_array = cv2.imread(filepath)#, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    cv2.imshow('image', new_array)
    cv2.waitKey(0)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = load_model(MODELDIR + MODEL)


for category in CATEGORIES:
    print(category, ':')
    for i in range(20):
        image = load_img(f'{DATADIR}/valid/{category}/{i}.jpg', target_size=(IMG_SIZE, IMG_SIZE))
        input_arr = img_to_array(image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)

        # OLD
        # prediction = model.predict([prepare(f'{DATADIR}/valid/{category}/{i}.jpg')])


        prediction = np.argmax(prediction[0])
        print('  ', CATEGORIES[prediction])
        # print('  ', prediction[0])

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