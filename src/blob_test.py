import cv2
import numpy as np
from sklearn import cluster
import matplotlib.pylab as plt
# import tesseract

# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'/Users/Bryceson/Downloads/Chrome/tesseract-ocr-w64-setup-v5.0.0-alpha.20201127.exe'
# print(pytesseract.image_to_string(r'D:\examplepdf2image.png'))


def find_rotated_rects(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dice_rects = []

    for contour in contours:

        # Rotated
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        area = w * h
        if area > BOX_MIN_AREA: dice_rects.append(rect)

    return dice_rects

def find_bounding_rects(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dice_rects = []

    for contour in contours:

        # Bounding
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect

        area = w * h
        if area > BOX_MIN_AREA: dice_rects.append(rect)

    return dice_rects


def draw_rotated_rect(rects, im):
    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

    return im

def draw_bounding_rects(rects, im):
    for rect in rects:
        x, y, w, h = rect
        im = cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return im

def crop(im, rect):
    angle = rect[2]

    # Rotate image
    rows, cols = im.shape[0], im.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    im_rot = cv2.warpAffine(im, M, (cols, rows))

    # Rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # Crop
    im_crop = im_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(im_crop)
    plt.show()
    # cv2.imshow('cropped', im_crop)
    return im_crop




# Initialize video feed
TEST_VIDS = [
    '../images/dice-set-roll.avi',
    '../images/dice-set-roll-1.avi',
    '../images/dice-set-roll-2.avi',
    '../images/dice-set-roll-3.avi',
    '../images/dice-set-roll-4.avi',
]

TEST_PICS = [
    '../images/dice-set.jpg',
    '../images/dice-set-2.jpg',
    '../images/dice-set-flash.jpg',
    '../images/dice-set-top.jpg',
    '../images/dice-set-top-flash.jpg',
    '../images/dice-set-bottom.jpg',
    '../images/dice-set-bottom-flash.jpg',
]



def manipulate_image(im):
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    THRESHOLD = 125
    im = cv2.medianBlur(im, 3)
    ret, im = cv2.threshold(im, THRESHOLD, 255, cv2.THRESH_BINARY)
    # im = cv2.Canny(im, 150, 230)

    return im


def handle_last_frame(raw):
    # Crop to image of each die
    im = manipulate_image(raw)


def old_blob_test(im):
    # Set up blob detector
    params = cv2.SimpleBlobDetector_Params()

    # Params
    params.minThreshold = 10
    params.maxThreshold = 255
    #
    params.filterByArea = True
    params.minArea = 10000
    params.maxArea = 1000000
    #
    # params.filterByCircularity = True
    # params.minCircularity = 0.01
    # params.maxCircularity = 1
    #
    # params.filterByConvexity = True
    # params.minConvexity = 0.01
    # params.maxConvexity = 1
    #
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    # params.maxInertiaRatio = 1


    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(im)
    # im = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # print(len(keypoints))



BOX_MIN_AREA = 100000



# IMAGE
raw = cv2.imread(TEST_PICS[1], cv2.IMREAD_GRAYSCALE)

im = manipulate_image(raw)
rects = find_bounding_rects(im)
# raw = draw_bounding_rects(rects, raw)
cv2.imshow('whole', raw)

cv2.waitKey(0)

i = 0
for rect in rects:
    path = '../gen/die' + str(i) + '.jpg'

    x, y, w, h = rect
    die = cv2.imread(TEST_PICS[1])[y:y+h, x:x+w]
    cv2.imshow('cropped', die)
    cv2.imwrite(path, die)
    cv2.waitKey(0)

    i+=1


exit(0)



# VIDEO
cap = cv2.VideoCapture(TEST_VIDS[0])
# cap = cv2.VideoCapture(0)
while True:
    # Grab the latest image from video feed
    ret, raw = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, -1)
        ret, raw = cap.read()

        handle_last_frame(raw)

    im = manipulate_image(raw)
    rects = find_rotated_rects(im)
    raw = draw_rotated_rect(rects, raw)
    cv2.imshow('frame', raw)

    res = cv2.waitKey(1)

    # # Stop if user presses 'q'
    if res & 0xFF == ord('q'): break


# Clean up when everything is finished
cap.release()
cv2.destroyAllWindows()