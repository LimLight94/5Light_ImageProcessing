import os
import cv2
import numpy
import logging

logger = logging.getLogger("main")


def is_cv2():
    major, minor, increment = cv2.__version__.split(".")
    return major == "2"


def is_cv3():
    major, minor, increment = cv2.__version__.split(".")
    return major == "3"

def is_cv4():
    major, minor, increment = cv2.__version__.split(".")
    return major == "4"


def display(title, img, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)

def display_red(title, img, location, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    img1 = img.copy()
    scale = numpy.sqrt(min(1.0, float(max_size) / (img1.shape[0] * img1.shape[1])))
    shape = (int(scale * img1.shape[1]), int(scale * img1.shape[0]))
    img1[location[0]:location[1], location[2]] = [0, 0, 255]
    img1[location[0]:location[1], location[3] - 1] = [0, 0, 255]
    img1[location[0], location[2]:location[3]] = [0, 0, 255]
    img1[location[1] - 1, location[2]:location[3]] = [0, 0, 255]
    img1 = cv2.resize(img1, shape)
    cv2.imshow(title, img1)

def save_image(path, result):
    name, ext = os.path.splitext(path)
    img_path = '{0}.png'.format(name)
    logger.debug('writing image to {0}'.format(img_path))
    cv2.imwrite(img_path, result)
    logger.debug('writing complete')
