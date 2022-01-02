import cv2 as cv
import numpy as np


def process(frame, dilate=0, erode=0, low_H=0, low_S=0, low_V=0, high_H=0, high_S=0, high_V=0):
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_processed = cv.erode(frame_HSV, np.ones((5, 5)), iterations=erode)
    frame_processed = cv.dilate(frame_processed, np.ones((5, 5)), iterations=dilate)
    frame_threshold = cv.inRange(frame_processed, (low_H, low_S, low_V), (high_H, high_S, high_V))

    return frame_threshold


# def find_contours(frame):
#
