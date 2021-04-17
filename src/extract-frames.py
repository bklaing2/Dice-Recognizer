import cv2

VIDS = [
    '../ml-data/dice-set/d6-1.avi',
    '../ml-data/dice-set/d6-2.avi',
    '../ml-data/dice-set/d20-1.avi',
    '../ml-data/dice-set/d20-2.avi',
]


# Save each frame as image
for i in range (2):
    cap = cv2.VideoCapture(VIDS[i+2])

    success, frame = cap.read()
    count = 0

    while success:
        cv2.imwrite(f'../ml-data/dice-set/d20/{i}-{count}.jpg', frame)
        success, frame = cap.read()
        count += 1


# Clean up when everything is finished
cap.release()
cv2.destroyAllWindows()