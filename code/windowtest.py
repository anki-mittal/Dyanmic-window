import cv2
import numpy as np


def callback(x):
    pass
# Read the image
image = cv2.imread('/home/pear/AerialRobotics/Aerial/HW5/allwindows/src/outputs/dynamicwindow/processed_frames/processed_0.png')
image = cv2.resize(image,(0,0), fx = 0.25, fy= 0.25)
cv2.namedWindow('image')


# Convert BGR to HSV
cv2.createTrackbar('Hue Min','image',0,179,callback)
cv2.createTrackbar('Hue Max','image',179,179,callback)
cv2.createTrackbar('Sat Min','image',0,255,callback)
cv2.createTrackbar('Sat Max','image',255,255,callback)
cv2.createTrackbar('Val Min','image',0,255,callback)
cv2.createTrackbar('Val Max','image',255,255,callback)

while(1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get current positions of the trackbars
    h_min = cv2.getTrackbarPos('Hue Min','image')
    h_max = cv2.getTrackbarPos('Hue Max','image')
    s_min = cv2.getTrackbarPos('Sat Min','image')
    s_max = cv2.getTrackbarPos('Sat Max','image')
    v_min = cv2.getTrackbarPos('Val Min','image')
    v_max = cv2.getTrackbarPos('Val Max','image')

    # Set the pink color range
    lower_pink = np.array([h_min, s_min, v_min])
    upper_pink = np.array([h_max, s_max, v_max])

    # Threshold the HSV image to get only pink colors
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)

    # Display the resulting frame
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()