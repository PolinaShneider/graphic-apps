import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np

img_original_path = 'p6.jpg'
img_specified_path = 'p5.jpg'


# calculate histogram
def calc_hist(img) -> list:
    hist = [0] * 256

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1

    return hist


# histogram equalization
def hist_equalize(source_image) -> list:
    accum = [0] * 256

    hist = calc_hist(source_image)
    size = source_image.shape[0] * source_image.shape[1]

    for i in range(0, 256):
        accum[i] = hist[i] / size

    for i in range(1, 256):
        accum[i] += accum[i - 1]

    accum = [round(i * 255) for i in accum]

    return accum


# histogram matching
def hist_match(orig, spec):
    values = [0] * 256

    hist_original = hist_equalize(orig)
    hist_specified = hist_equalize(spec)

    res = copy.deepcopy(orig)

    for i in range(0, 256):
        j = 255
        while True:
            values[i] = j
            j = j - 1
            if j < 0 or hist_original[i] > hist_specified[j]:
                break

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i][j] = values[res[i][j]]

    return res


# read images
original = cv2.imread(img_original_path, 0)
specified = cv2.imread(img_specified_path, 0)

# plot original histograms
plt.figure(1)
plt.plot(calc_hist(original))

plt.figure(2)
plt.plot(calc_hist(specified))

# plot matching histogram result
plt.figure(3)
plt.plot(calc_hist(hist_match(original, specified)))

plt.show()

# show original images
combined = np.hstack((original, specified))
cv2.imshow("Histogram matching", combined)

# show matching image result
result = hist_match(original, specified)
cv2.imshow("Result of matching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

