import cv2
import numpy as np


def add_tuple(x, y):
    return x[0] + y[0], x[1] + y[1]


def get_gray_color(cl):
    return cv2.cvtColor(np.array([[cl]], dtype=np.uint8), cv2.COLOR_RGB2GRAY)[0, 0]


