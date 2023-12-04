import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('jjk.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Primera ventana: Imagen original (input)
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title('Input')
plt.axis('off')
plt.show()

# Segunda ventana: Erosión y dilatación
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(erosion, cmap='gray')
ax[0].set_title('Erosion')
ax[0].axis('off')

ax[1].imshow(dilation, cmap='gray')
ax[1].set_title('Dilation')
ax[1].axis('off')

plt.tight_layout()
plt.show()

# Tercera ventana: Apertura y cierre
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(opening, cmap='gray')
ax[0].set_title('Opening')
ax[0].axis('off')

ax[1].imshow(closing, cmap='gray')
ax[1].set_title('Closing')
ax[1].axis('off')

plt.tight_layout()
plt.show()
