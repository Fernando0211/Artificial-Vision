import cv2
import numpy as np

image = cv2.imread('spider.jpg', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('spider.jpg', cv2.IMREAD_COLOR)

_, binary_image = cv2.threshold(image, 45, 500, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)

#Expands white regions
dilated = cv2.dilate(binary_image, kernel, iterations=1)

#Shrinks white regions
eroded = cv2.erode(binary_image, kernel, iterations=1)

#Combines dilation and erosion to close gaps while preserving shape
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

#Combines erosion and dilation to remove small white objects
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Find edges and draw object boundaries
boundary_extraction = cv2.subtract(binary_image, cv2.erode(binary_image, kernel))
canny_image = cv2.Canny(boundary_extraction, 50, 150)
contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)  # Draw contours in blue

# Applies dilation to make white regions thicker
thickened_image = cv2.dilate(binary_image, kernel, iterations=1)

#  Applies erosion to reduce white regions' thickness
thinned_image = cv2.erode(binary_image, kernel, iterations=1)

cv2.imshow('Input', binary_image)
cv2.imshow('Dilated', dilated)
cv2.imshow('Eroded', eroded)
cv2.imshow('Closing', closing)
cv2.imshow('Opening', opening)
cv2.imshow('Boundary Extraction', img_color)
cv2.imshow('Thickened', thickened_image)
cv2.imshow('Thinned', thinned_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
