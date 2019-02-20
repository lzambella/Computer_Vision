import cv2 as cv
import numpy as np


class GameBoard:

    """
    Class for handing game board detection from either an image or feed
    Utimately the class will contain data structures as to where each space on
    a standard game board are located
    """

    def __init__(self, image_path):
        """
        Initialize a new board detection process using a static image
        :param image_path:
        """
        self.cv_image = cv.imread(image_path)

    def __init__(self, camera_index):
        """
        Initialize a new board detection object using a camera feed instead of a static image
        :param camera_index: index of the video feed
        :return:
        """
        self.webcam = cv.VideoCapture(camera_index)

    def grab_frame(self):
        """
        Grab a single frame from the camera
        :return: Frame as numpy umat
        """
        return self.webcam.read()

