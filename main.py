import math

import cv2 as cv
import toml

from process import *

configs = toml.load("config.toml")

# real world dimensions of the goal target
# These are the full dimensions around both strips
TARGET_STRIP_LENGTH = 19.625  # inches
TARGET_HEIGHT = 17.0  # inches
TARGET_TOP_WIDTH = 39.25  # inches
TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2 * TARGET_STRIP_LENGTH * math.cos(math.radians(60))

# [0, 0] is center of the quadrilateral drawn around the high goal target
# [top_left, bottom_left, bottom_right, top_right]
real_world_coordinates = np.array([
    [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
    [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
    [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
    [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
])


camera_matrix = np


def sort_corners(contour, center=None):
    """Sort the contour in our standard order, starting upper-left and going counter-clockwise"""

    # Note: the inputs are all numpy arrays, so it is fast to operate on the whole array at once
    if center is None:
        center = contour.mean(axis=0)

    d = contour - center
    # remember that y-axis increases down, so flip the sign
    angle = (np.arctan2(-d[:, 1], d[:, 0]) - (math.pi / 2)) % (2 * math.pi)
    return contour[np.argsort(angle)]


if __name__ == '__main__':
    while True:
        frame = cv.imread("data/test_images/2020/wpi_sample_images/BlueGoal-060in-Center.jpg")

        processed = process(frame, configs)
        frame_threshold = threshold(processed, configs)
        gray = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
        edges = find_edges(frame_threshold, configs)
        contours = find_contours(edges)

        # Only look at the 5 largest contours
        contour_list = []
        for c in contours:
            center, widths = contour_center_width(c)
            area = widths[0] * widths[1]
            if area > configs["filtering"]["min_area"]:
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        contour_list.sort(key=lambda c: c['area'], reverse=True)

        contour = contour_list[0]

        outer_corners = sort_corners(contour['contour'], contour['center'])
        retval, rvec, tvec = cv.solvePnP(real_world_coordinates, outer_corners,
                                         cameraMatrix, distortionMatrix)

        cv.drawContours(frame, [contour["contour"]], -1, (0, 255, 0), 3)
        cv.imshow("frame", frame)
        cv.imshow("processed", frame_threshold)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
