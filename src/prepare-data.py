import random
import os
import pickle
import numpy as np
import cv2

DATADIR = '../ml-data/dice-set/train'
CATEGORIES = ['d12', 'd4']

IMG_SIZE = 128





def create_training_data():
    training_data = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])

            except Exception as e: pass

    random.shuffle(training_data)

    return training_data


# Create data
training_data = create_training_data()

print(len(training_data))
for sample in training_data[:10]: print(sample[1])


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


# DON'T FULLY UNDERSTAND WHY THIS IS NEEDED
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1 is because it's grayscale (need 3 for color)


# Save data
with open(DATADIR + '/X.pickle', 'wb') as pkl:
    pickle.dump(X, pkl)

with open(DATADIR + '/y.pickle', 'wb') as pkl:
    pickle.dump(y, pkl)


# Load data
# X = pickle.load(open(DATADIR + '/X.pickle', 'rb'))
# y = pickle.load(open(DATADIR + '/y.pickle', 'rb'))

print(X[1])
print(y[1])