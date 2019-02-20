import cv2 as cv
import numpy as np


print("open webcam")
webcam = cv.VideoCapture(1)
detector = cv.SimpleBlobDetector()

while True:
    ret, frame = webcam.read()
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayscale = cv.blur(grayscale, (3,3))
    threshold = cv.adaptiveThreshold(grayscale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    threshold = cv.bitwise_not(threshold)

    # houghlines = cv.HoughLinesP(threshold, 1, 3.14159, 200, minLineLength=100, maxLineGap=50)
    # Detect the board (Should have a white outline)
    contours, hier = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # fInd largest object with four corners
    area = cv.contourArea(contours[0])
    cont = contours[0]
    for contour in contours:
        loc = cv.contourArea(contour)
        if loc > area:
            houghlines = cv.HoughLinesP(loc, 1, 3.14159, 200, maxLineGap=50, minLineLength=100)
            for rho, theta in houghlines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            area = cv.contourArea(contour)
            cont = contour

    cv.drawContours(frame, cont, -1, (0,255,0), thickness=2)
    thresh_color = cv.cvtColor(threshold,cv.COLOR_GRAY2BGR)
    cv.drawContours(thresh_color, contours, -1, (0,255,0), thickness=2)
    cv.imshow("press q to quit", frame)
    cv.imshow("a", thresh_color)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()

