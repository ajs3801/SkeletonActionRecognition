import cv2
import numpy as np

src = "/Users/jaejoon/LGuplus/main_project/lstm/videos/original_videos/lunge-down/01-202208041440-0-20-0.avi"
# Turn on Laptop's webcam
cap = cv2.VideoCapture(src)

while True:

    ret, frame = cap.read()

    h, w, c = frame.shape
    print(frame.shape)

    # Locate points of the documents
    # or object which you want to transform
    pts1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    pts2 = np.float32([[0, 0], [0, h], [w, h - 100], [w, 100]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (w, h))

    # Wrap the transformed image
    # cv2.imshow("frame", frame)  # Initial Capture
    cv2.imshow("frame1", result)  # Transformed Capture

    if cv2.waitKey(24) == 27:
        break

cap.release()
cv2.destroyAllWindows()
