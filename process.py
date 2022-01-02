import cv2 as cv
import numpy as np


def process(frame, config):
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_processed = cv.erode(frame_HSV, np.ones((5, 5)), iterations=config['erode'])
    frame_processed = cv.dilate(frame_processed, np.ones((5, 5)), iterations=config['dilate'])
    frame_threshold = cv.inRange(frame_processed,
                                 np.array(config['lower_HSV']),
                                 np.array(config['upper_HSV']))

    return frame_threshold


# def find_contours(frame):
#
