import cv2
import matplotlib.pyplot as plt  # pip install matplotlib
import numpy as np

image_path = "images/1.jpg"

img = cv2.imread(image_path)


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def remove_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = hsv[..., 2]

    thresh_val, _ = cv2.threshold(
        brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.threshold(brightness, thresh_val, 255, cv2.THRESH_BINARY)[1]

    result = cv2.bitwise_and(image, image, mask=mask)

    return result


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


image = cv2.imread(image_path)

gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)
background = remove_background(image)

fig, axs = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(12, 12))

ax1, ax2 = axs[0]
ax3, ax4 = axs[1]
ax5, ax6 = axs[2]

ax1.title.set_text("Gray")
ax1.imshow(gray, cmap="gray")

ax2.title.set_text("Thresh")
ax2.imshow(thresh, cmap="gray")

ax3.title.set_text("Opening")
ax3.imshow(opening, cmap="gray")

ax4.title.set_text("Canny")
ax4.imshow(canny, cmap="gray")

ax5.title.set_text("Original")
ax5.imshow(image, cmap="gray")

ax6.title.set_text("Without Background")
ax6.imshow(background, cmap="gray")

plt.show()
