import cv2 as cv
import numpy as np


def process(frame, config):
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_processed = cv.erode(frame_HSV, np.ones((5, 5)), iterations=config['erode'])
    frame_processed = cv.dilate(frame_processed, np.ones((5, 5)), iterations=config['dilate'])

    return frame_processed


def threshold(frame, config):
    frame_threshold = cv.inRange(frame,
                                 np.array(config['lower_HSV']),
                                 np.array(config['upper_HSV']))
    return frame_threshold


def find_contours(frame):
    contours, _ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def find_edges(frame, config):
    blur = cv.blur(frame, (config['blur'], config['blur']))
    edges = cv.Canny(blur, config['threshold1'], config['threshold2'])
    return edges
