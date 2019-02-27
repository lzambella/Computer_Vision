'''
This class is preliminary experimentation before tidying things up into their own functions and classes
'''

import cv2 as cv
import numpy as np
from time import sleep


def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    '''
    Determine the intersection of two lines given only the endpoints
    :return: Coordinates of the lines, if any, in tuple form. It also returns t and u in tuple form.
    if both t and u are between 0 and 1.0, then the intersection exists between the two line segments.
    '''

    Px_a = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
    Px_b = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    Px = Px_a/Px_b

    Py_a = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)
    Py_b = Px_b
    Py = Py_a/Py_b

    t_u = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
    t_d = (x1-x2)*(y3-y4)-(y1-y2)*(x3*x4)

    u_u = (x1-x2)*(y1-y3)-(y1-y2)*(x1-x3)
    u_d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)

    u = -1*(u_u/u_d)
    t = t_u/t_d

    return (Px, Py), (t,u)


def draw_lines(src, lines):
    '''
    Draws all lines from a HoughLineP transform onto a source image
    :param src: 
    :param lines: 
    :return: 
    '''
    print len(lines[0])
    if lines is None or len(lines) == 0:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 4)


def draw_point(src, coord):
    """
    Draws a circular point centered of coord
    :param src: Source image to draw on
    :param coord: coordinate to point towards as a tuple
    :return:
    """
    cv.circle(src, coord, 4, (0, 255, 0))


def grab_frame(webcam_source):
    """
    Grabs and returns a single frame from a webcam source
    """
    ret, f = webcam_source.read()
    size = f.shape

    # if the image is too large, resize it
    if size[0] * size[1] > 10 ** 6:
        f = cv.resize(f, (0, 0), fx=0.1, fy=0.1)
    return f


def image_threshold(image_matrix):
    '''
    Takes in a numPY matrix as a image and returns the threshold of that image
    :return:
    '''
    grayscale = cv.cvtColor(image_matrix, cv.COLOR_BGR2GRAY)
    grayscale = cv.blur(grayscale, (2,2))
    threshold = cv.adaptiveThreshold(grayscale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
    threshold = cv.bitwise_not(threshold)
    return threshold


def find_center(contour):
    """
    Finds the centerpoint of a closed loop contour
    :param contour:
    :return: (x,y) corrdinates of the centerpoint
    """
    moments = cv.moments(contour)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    return cX, cY


def get_contours(image_matrix):
    '''
    Gets the contours in an image matrix
    :return:
    '''
    # Detect the board (Should have a white outline)
    contours, hier = cv.findContours(image_matrix, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours


def sort_contours(contours):
    """
    Sorts a set of contours from smallest to largest
    :param contours:
    :return:
    """
    contours_sorted = sorted(contours, key=lambda x: cv.contourArea(x))
    return contours_sorted


def get_approx_sides(contour):
    """
    Gets the approximate number of sides a contour shape has
    assumes that the shape is closed
    :param contour:
    :return:
    """
    approx = cv.approxPolyDP(contour, 0.5 * cv.arcLength(contour, True), True)
    return len(approx)


def get_largest_square(contours):
    """
    Gets the largest 4 sided contour in a set
    Used to detect a game board
    :param contours: set of contours as returned by cv.findContours
    :return: Largest 4 sided shape
    """
    cnts_sorted = sort_contours(contours)
    arr_len = len(cnts_sorted) - 1
    for x in range(arr_len, 0):
        contour = cnts_sorted[x]
        sides = get_approx_sides(contour)
        if sides == 4:
            return contour

    # Return the largest contour if no rectangles have been found as a failsafe
    return cnts_sorted[arr_len]


def contour_to_mask(contour, frame):
    """
    Takes a contour, fills it, and then creates an image mask out of it
    :param contour: Contour to create a mask from
    :param frame: Frame to reference when created the mask
    :return:
    """
    # First create an empty mask with the width and height of the image frame
    mask = np.zeros(frame.shape, dtype='uint8')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    cv.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
    cv.drawContours(frame, [contour], -1, (0, 0, 255), thickness=3)
    frame_masked = cv.bitwise_and(frame, frame, mask=mask)
    return frame_masked


def quick_canny(image):
    """
    Quicky convert frame to black and white and then edge detect
    :param image:
    :return:
    """
    bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(bw, 50, 150)
    return canny


def get_lines(image):
    """
    Performs a HoughLineP transform on the specified image
    :param image:
    :return:
    """
    canny = quick_canny(image)
    cv.imshow("canny", canny)
    lines = cv.HoughLinesP(canny, 1, 3.14159 / 180, 50,  minLineLength=100, maxLineGap=100)
    return lines


def find_grid(image):
    """
    Method to find a grid on an image
    normally takes in a masked image of a chess board
    :param image: numpy matrix with image data
    :return: coordinates of all intersections in image
    """
    # Find all lines in the image
    lines = get_lines(image)


def find_chess_board(image_source):
    """
    Finds the contour for the chessboard in an image
    assumes the chessboard is the largest square object in the frame
    :param image_source:
    :return:
    """
    thresh = image_threshold(image_source)

    # Find all the contours in the image and then
    # find largest object with four corners
    conts = get_contours(thresh)
    square_contour = get_largest_square(conts)
    return square_contour


def main():
    print("open webcam")
    webcam = cv.VideoCapture(0)
    while True:
        frame = grab_frame(webcam)

        chess_board = find_chess_board(frame)
        # Create a mask for the largest 4 sided contour
        masked = contour_to_mask(chess_board, frame)

        center_point = find_center(chess_board)
        draw_point(frame, center_point)
        cv.imshow("press q to quit", masked)
        cv.imshow("camera feed", frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
