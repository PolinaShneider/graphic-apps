from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./img3.jpg"


# calculate histogram
def calc_hist(img) -> list:
    hist = [0] * 256

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1

    return hist


# equalization function
def hist_equalize(source_image):
    modified_image = deepcopy(source_image)
    hist = calc_hist(modified_image)
    size = modified_image.shape[0] * modified_image.shape[1]

    accum = [0] * 256

    for i in range(0, 256):
        accum[i] = hist[i] / size

    for i in range(1, 256):
        accum[i] += accum[i - 1]

    for i in range(modified_image.shape[0]):
        for j in range(modified_image.shape[1]):
            modified_image[i][j] = round(accum[modified_image[i][j]] * 255)

    return modified_image


# perform equalization
image_before = cv2.imread(img_path, 0)
hist_before = calc_hist(image_before)
image_after = hist_equalize(image_before)
hist_after = calc_hist(image_after)

plt.figure(1)
plt.plot(hist_before, color='g')

plt.figure(2)
plt.plot(hist_after, color='r')

# show hists on plot
plt.show()

# show images
combined = np.hstack((image_before, image_after))
cv2.imshow("Histogram equalization", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

