from __future__ import print_function
import cv2 as cv
import argparse
from process import *
import toml

import numpy as np

config = toml.load('config.toml')
hsv_config = config["preprocessing"]
canny_config = config['canny']

max_value = 255
max_value_H = 360 // 2
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global hsv_config
    hsv_config['lower_HSV'][0] = min(hsv_config['upper_HSV'][0] - 1, val)
    cv.setTrackbarPos(low_H_name, window_detection_name, hsv_config['lower_HSV'][0])


def on_high_H_thresh_trackbar(val):
    global hsv_config
    hsv_config['upper_HSV'][0] = max(hsv_config['lower_HSV'][0] + 1, val)
    cv.setTrackbarPos(high_H_name, window_detection_name, hsv_config['upper_HSV'][0])


def on_low_S_thresh_trackbar(val):
    global hsv_config
    hsv_config['lower_HSV'][1] = min(hsv_config['upper_HSV'][1] - 1, val)
    cv.setTrackbarPos(low_S_name, window_detection_name, hsv_config['lower_HSV'][1])


def on_high_S_thresh_trackbar(val):
    global hsv_config
    hsv_config['upper_HSV'][1] = max(hsv_config['lower_HSV'][1] + 1, val)
    cv.setTrackbarPos(high_S_name, window_detection_name, hsv_config['upper_HSV'][1])


def on_low_V_thresh_trackbar(val):
    global hsv_config
    hsv_config['lower_HSV'][2] = min(hsv_config['upper_HSV'][2] - 1, val)
    cv.setTrackbarPos(low_V_name, window_detection_name, hsv_config['lower_HSV'][2])


def on_high_V_thresh_trackbar(val):
    global hsv_config
    hsv_config['upper_HSV'][2] = max(hsv_config['lower_HSV'][2] + 1, val)
    cv.setTrackbarPos(high_V_name, window_detection_name, hsv_config['upper_HSV'][2])


def on_erosion(val):
    global hsv_config
    hsv_config['erode'] = val
    cv.setTrackbarPos("Erosion", window_detection_name, val)


def on_dilation(val):
    global hsv_config
    hsv_config['erode'] = val
    cv.setTrackbarPos("Dilation", window_detection_name, val)


def on_canny1(val):
    global canny_config
    canny_config['threshold1'] = val
    cv.setTrackbarPos("Canny 1", window_detection_name, val)


def on_canny2(val):
    global canny_config
    canny_config['threshold2'] = val
    cv.setTrackbarPos("Canny 2", window_detection_name, val)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, hsv_config['lower_HSV'][0], max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, hsv_config['upper_HSV'][0], max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, hsv_config['lower_HSV'][1], max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, hsv_config['upper_HSV'][1], max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, hsv_config['lower_HSV'][2], max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, hsv_config['upper_HSV'][2], max_value, on_high_V_thresh_trackbar)
cv.createTrackbar("Erosion", window_detection_name, hsv_config['erode'], 2, on_erosion)
cv.createTrackbar("Dilation", window_detection_name, hsv_config['dilate'], 2, on_dilation)
cv.createTrackbar("Canny 1", window_detection_name, canny_config['threshold1'], 500, on_canny1)
cv.createTrackbar("Canny 2", window_detection_name, canny_config['threshold2'], 500, on_canny2)
while True:

    ret, frame = cap.read()
    if frame is None:
        break

    processed = process(frame, hsv_config)
    frame_threshold = threshold(processed, hsv_config)
    contours = find_contours(frame_threshold)
    gray = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
    edges = find_edges(gray, canny_config)
    edges1 = find_edges(frame_threshold, canny_config)

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    cv.imshow("Edges", edges)
    cv.imshow("Edges 2", edges1)


    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
