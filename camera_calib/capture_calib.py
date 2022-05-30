import cv2
import numpy as np

# Returns video from the first webcam on your computer
cap = cv2.VideoCapture(0)

img_ctr = 1
while(True):
    # success - boolean that indicates if frame is read correctly
    success, frame = cap.read()

    # Flips the frame
    frame = cv2.flip(frame, 1)

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    foundChessboardCorners, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if foundChessboardCorners:
      cv2.imwrite(f'test{img_ctr}.jpg', frame)
      img_ctr += 1
      print(f"{img_ctr} images obtained!")
    # Shows the frame
    cv2.imshow('Frame', frame)

    # Allows the window to be quitted out by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break