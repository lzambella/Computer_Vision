'''
This class is preliminary experimentation before tidying things up into their own functions and classes
'''

import cv2 as cv
import numpy as np


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
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except TypeError:
        print "No lines detected"


print("open webcam")
# webcam = cv.VideoCapture(1)
detector = cv.SimpleBlobDetector()

while True:
    # ret, frame = webcam.read()
    frame = cv.imread("chess2.jpg")
    size = frame.shape

    # if the image is too large, resize it
    if size[0] * size[1] > 10**6:
        frame = cv.resize(frame, (0,0), fx=0.1, fy=0.1)

    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayscale = cv.blur(grayscale, (3,3))
    threshold = cv.adaptiveThreshold(grayscale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    threshold = cv.bitwise_not(threshold)

    # houghlines = cv.HoughLinesP(threshold, 1, 3.14159, 200, minLineLength=100, maxLineGap=50)
    # Detect the board (Should have a white outline)
    contours, hier = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


    # find largest object with four corners
    area = cv.contourArea(contours[0])
    cont = contours[0]
    for contour in contours:
        loc = cv.contourArea(contour)
        approx = cv.approxPolyDP(contour, 0.1 * cv.arcLength(contour, True), True)

        # Find the largest square contour
        if loc > area and len(approx) == 4:
            area = cv.contourArea(contour)
            cont = contour

    # Create a mask for the largest 4 sided contour
    mask = np.zeros(frame.shape, dtype='uint8')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    cv.drawContours(mask, [cont], -1, (255, 255, 255), thickness=cv.FILLED)

    cv.drawContours(frame, [cont], -1, (0, 0, 255), thickness=3)

    frame_masked = cv.bitwise_and(frame, frame, mask=mask)

    # perform hough lines on the masked image to find grid
    frameMaskedGray = cv.cvtColor(frame_masked, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(frameMaskedGray, 100, 150, apertureSize=3)

    lines = cv.HoughLinesP(edges, 1, 3.14159/180, 50, minLineLength=0, maxLineGap=100)
    draw_lines(frame, lines)

    # Create a new image using the four corners of the detected chess board
    # Start by remaking the contour mask
    mask = cv.Canny(mask,50,255)
    lines = cv.HoughLinesP(mask, 1, 3.14/720, 10, minLineLength=5, maxLineGap=300)
    draw_lines(frame_masked, lines)

    # Create our numpy array for storing points in an image
    point_coords = []
    for i in range(1, len(lines)):
        intersect, constr = intersection(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3], lines[i-1][0][0], lines[i-1][0][1], lines[i-1][0][2], lines[i-1][0][3])
        if intersect == 0:
            continue
        else:
            point_coords.append(intersect)

    for coord in point_coords:
        cv.circle(frame,coord,3,(255,0,0),-1)

    cv.imshow("press q to quit", frame_masked)
    cv.imshow("b", frame)
    cv.imshow("a", mask)
    cv.imshow('c', edges)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# webcam.release()
cv.destroyAllWindows()

