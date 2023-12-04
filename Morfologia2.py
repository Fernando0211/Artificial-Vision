import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'fanuc.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(path, cv2.IMREAD_COLOR)

img_copy = img.copy()
img_copy = cv2.GaussianBlur(img, (3, 3), 0)

edges = cv2.Canny(img_copy, 80, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_color, contours, -1, (255, 0, 255), 1)

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
fill = img - erosion

boundary = fill
boundary0 = 255 - boundary

seed = np.zeros_like(boundary0)
seed[506, 506] = 255

kernel1 = np.ones((3, 3), np.uint8)

while True:
    dilated = cv2.dilate(seed, kernel1, iterations=1)
    fill_img = cv2.bitwise_and(dilated, boundary0)
    if np.array_equal(fill_img, seed):
        break
    seed = fill_img.copy()

seed = 255 - seed
fill_img = boundary | seed

binary = cv2.adaptiveThreshold(img_copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cleaned = np.ones_like(binary) * 255
cv2.drawContours(cleaned, contours, -1, (0), thickness=cv2.FILLED)
cleaned = 255 - cleaned
binary = cv2.erode(cleaned, kernel, iterations=1)
binary = cv2.dilate(binary, kernel, iterations=1)

num_labels, labels = cv2.connectedComponents(binary)

output = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
colors = []
for i in range(1, num_labels):
    colors.append(list(np.random.randint(0, 255, 3)))

for y in range(output.shape[0]):
    for x in range(output.shape[1]):
        if labels[y, x] > 0:
            output[y, x, :] = colors[labels[y, x] - 1]

# Visualizar imágenes con matplotlib
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].set_title('Input')
ax[0, 0].axis('off')

ax[0, 1].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
ax[0, 1].set_title('Boundary')
ax[0, 1].axis('off')

ax[1, 0].imshow(fill_img, cmap="gray")
ax[1, 0].set_title('Fill')
ax[1, 0].axis('off')

ax[1, 1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[1, 1].set_title('Connected Components')
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()
