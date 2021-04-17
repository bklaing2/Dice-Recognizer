import random
import os
from shutil import copyfile
import pickle
import numpy as np
import cv2

DATADIR = '../ml-data/dice-set/train'
CATEGORIES = ['d6', 'd20']


for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    os.mkdir(path + '/old-names')

    class_num = CATEGORIES.index(category)
    count = 0

    for img in os.listdir(path):
        try:
            file_path = os.path.join(path, img)
            copyfile(file_path, path + '/old-names/' + img)

            os.rename(file_path, path + '/' + str(count) + '.jpg')
            count += 1

        except Exception as e: pass