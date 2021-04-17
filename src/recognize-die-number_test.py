import cv2
import numpy as np
from sklearn import cluster

THRESHOLD = 75


# Set up blob detector
params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.6

detector = cv2.SimpleBlobDetector_create(params)

def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)

    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

    blobs = detector.detect(frame_gray)
    return blobs



def get_dice_from_blobs(blobs):
    # Get centroids of all blobs
    X = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        # Important to set min_sample to 0, as a dice-set may only have one dot
        clustering = cluster.DBSCAN(eps=40, min_samples=0).fit(X)

        # Find the largest label assigned + 1, that's the number of dice-set found
        num_dice = max(clustering.labels_) + 1
        print(num_dice)

        dice = []

        # Calculate centroid of each dice-set, the average between all a dice-set's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame, dice, blobs):
    # Overlay blobs
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice-set number
    for d in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)




def find_bounds(frame):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dice_rects = 0

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        # if w != 0 and h != 0:
        #     aspect = abs(w/h)
        #     if aspect < 1:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    return frame, dice_rects


def extract_dice(frame):
    pass


def get_die_number(die):
    pass


# Initialize video feed
TEST_VIDS = [
    '../images/dice-set-roll.avi',
    '../images/dice-set-roll-1.avi',
    '../images/dice-set-roll-2.avi',
    '../images/dice-set-roll-3.avi',
    '../images/dice-set-roll-4.avi',
]

TEST_PICS = [
    '../images/dice-set-top.jpg',
    '../images/dice-set-top-flash.jpg',
    '../images/dice-set-bottom.jpg',
    '../images/dice-set-bottom-flash.jpg',
]


cap = cv2.VideoCapture(TEST_VIDS[0])
THRESHOLD = 100
while True:
    # Grab the latest image from video feed
    ret, frame = cap.read()


    if not ret:
        THRESHOLD = int(input('Threshold: '))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    frame_blurred = cv2.medianBlur(frame, 3)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

    ret, frame_thresh = cv2.threshold(frame_gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    frame_edges = cv2.Canny(frame_thresh, 80, 230)
    frame_bounds, num_dice = find_bounds(frame_thresh)

    cv2.imshow('frame', frame_bounds)
    # print(num_dice)

    # # Define these later
    # blobs = get_blobs(frame)
    # dice-set = get_dice_from_blobs(blobs)
    # out_frame = overlay_info(frame, dice-set, blobs)
    #
    # cv2.imshow('frame', frame_bounds)


    #
    res = cv2.waitKey(1)
    #
    # # Stop if user presses 'q'
    if res & 0xFF == ord('q'): break


# Clean up when everything is finished
cap.release()
cv2.destroyAllWindows()