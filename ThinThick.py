import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'jjk.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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

def thicking(img): #fondo negro
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    imgW = cv2.bitwise_not(img) #fonde blanco
    dilation = cv2.dilate(img, kernel, iterations=1)
    intersection = cv2.bitwise_and(imgW, dilation)
    thick = cv2.bitwise_or(img, intersection)

    return thick

def preprocessing(img):

    kernel = np.ones((3, 3), np.uint8)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    pimg = cv2.bitwise_not(closing)
    
    return(pimg)

image = preprocessing(img)
thin = thinning(image)
thick = thicking(image)

# Visualizar la imagen de entrada y la imagen binarizada
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img, cmap="gray")
ax[0].set_title('Input')
ax[0].axis('off')

ax[1].imshow(image, cmap="gray")
ax[1].set_title('Binary')
ax[1].axis('off')

plt.tight_layout()
plt.show()

# Visualizar la imagen después del adelgazamiento y engrosamiento
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(thin, cmap="gray")
ax[0].set_title('Thinning')
ax[0].axis('off')

ax[1].imshow(thick, cmap="gray")
ax[1].set_title('Thicking')
ax[1].axis('off')

plt.tight_layout()
plt.show()





