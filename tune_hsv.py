from __future__ import print_function
import cv2 as cv
import argparse
from process import process
import toml

import numpy as np


configs = toml.load("config.toml")["preprocessing"]

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
    global configs
    configs['lower_HSV'][0] = min(configs['upper_HSV'][0] - 1, val)
    cv.setTrackbarPos(low_H_name, window_detection_name, configs['lower_HSV'][0])


def on_high_H_thresh_trackbar(val):
    global configs
    configs['upper_HSV'][0] = max(configs['lower_HSV'][0] + 1, val)
    cv.setTrackbarPos(high_H_name, window_detection_name, configs['upper_HSV'][0])


def on_low_S_thresh_trackbar(val):
    global configs
    configs['lower_HSV'][1] = min(configs['upper_HSV'][1] - 1, val)
    cv.setTrackbarPos(low_S_name, window_detection_name, configs['lower_HSV'][1])


def on_high_S_thresh_trackbar(val):
    global configs
    configs['upper_HSV'][1] = max(configs['lower_HSV'][1] + 1, val)
    cv.setTrackbarPos(high_S_name, window_detection_name, configs['upper_HSV'][1])


def on_low_V_thresh_trackbar(val):
    global configs
    configs['lower_HSV'][2] = min(configs['upper_HSV'][2] - 1, val)
    cv.setTrackbarPos(low_V_name, window_detection_name, configs['lower_HSV'][2])


def on_high_V_thresh_trackbar(val):
    global configs
    configs['upper_HSV'][2] = max(configs['lower_HSV'][2] + 1, val)
    cv.setTrackbarPos(high_V_name, window_detection_name, configs['upper_HSV'][2])


def on_erosion(val):
    global configs
    configs['erode'] = val
    cv.setTrackbarPos("Erosion", window_detection_name, val)


def on_dilation(val):
    global configs
    configs['erode'] = val
    cv.setTrackbarPos("Dilation", window_detection_name, val)


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, configs['lower_HSV'][0], max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, configs['upper_HSV'][0], max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, configs['lower_HSV'][1], max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, configs['upper_HSV'][1], max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, configs['lower_HSV'][2], max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, configs['upper_HSV'][2], max_value, on_high_V_thresh_trackbar)
cv.createTrackbar("Erosion", window_detection_name, configs['erode'], 2, on_erosion)
cv.createTrackbar("Dilation", window_detection_name, configs['dilate'], 2, on_dilation)
while True:

    ret, frame = cap.read()
    if frame is None:
        break

    frame_threshold = process(frame, configs)

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
