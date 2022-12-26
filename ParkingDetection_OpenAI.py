import cv2
import numpy as np

image = cv2.imread('parking2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

mask = np.zeros_like(gray)
cv2.drawContours(mask, [max_contour], -1, 255, -1)
cv2.drawContours(image, [max_contour], -1, 255, -1)
thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Image", image)
cv2.waitKey(0)

occupied = np.sum(np.logical_and(mask, thresholded))
if occupied > 1000:
    print("Parking spot is occupied")
else:
    print("Parking spot is not occupied")
