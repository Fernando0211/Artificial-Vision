import cv2 as cv2
import numpy as np

path = 'zebra.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(path, cv2.IMREAD_COLOR)

kernel = np.ones((3, 3), np.uint8)


def preprocessing(img):

    kernel = np.ones((3, 3), np.uint8)
    blur = cv2.GaussianBlur(img, (13, 13), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    pimg = cv2.bitwise_not(closing)
    
    return(pimg)

# Boundary extraction is achieved by subtracting the eroded set from the original foreground
def edgextraction(imgP, imgColor):
    edges = cv2.Canny(imgP, 80, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgColor, contours, -1, (255, 0, 255), 2)

    return(edges)

# Thinning iteratively erodes and dilates A, subtracting each from the original set,
# then combines the first erosion with the new subtracted set, resulting in thinned edges.
def thinning(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    thin = np.copy(img)
    prev = np.zeros(img.shape, dtype=img.dtype)
    
    while cv2.norm(thin - prev) != 0:
        prev = thin.copy()

        erosion = cv2.erode(thin, kernel)
        dilation = cv2.dilate(erosion, kernel)

        dilation = thin - dilation

        thin = cv2.bitwise_or(erosion, dilation)

    return thin

# Thickening is achieved by complementing set A, dilating the complement, and comparing it with the original A. 
# Pixels where foreground values coincide are added to a new array. 
# This new array, containing matched pixels from the dilation, is then added to the original A, resulting in thickened edges.
def thicking(img):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    imgW = cv2.bitwise_not(img)
    dilation = cv2.dilate(img, kernel, iterations=1)
    intersection = cv2.bitwise_and(imgW, dilation)
    thick = cv2.bitwise_or(img, intersection)

    return thick


imgP = preprocessing(img)

erosion = cv2.erode(imgP, kernel, iterations=1) # Erosion retains foreground in eroded image only if the entire structuring element is within foreground
dilation = cv2.dilate(imgP, kernel, iterations=1) # In dilation, the center of the structuring element becomes a foreground pixel if at least one of its pixels is foreground
opening = cv2.morphologyEx(imgP, cv2.MORPH_OPEN, kernel) # Opening is erosion followed by dilation, emphasizing major structures by removing smaller details
closing = cv2.morphologyEx(imgP, cv2.MORPH_CLOSE, kernel) # Closing is dilation followed by erosion
edges = edgextraction(imgP, img_color)
thin = thinning(imgP)
thick = thicking(imgP)


# Muestra el resultado
cv2.imshow('Input', imgP)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Edge', img_color)
cv2.imshow('Thinning', thin)
cv2.imshow('Thicking', thick)

cv2.waitKey(0)
cv2.destroyAllWindows()