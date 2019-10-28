import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

percent = 0.05
img_path = "./p5.jpg"

# linear stretching params
c = 0
d = 255


# function for stretching
def stretch(value):
    if value < a:
        value = a
    if value > b:
        value = b
    return ((value - a) * (d - c) / (b - a)) + c


# calculate new top and bottom limits of histogram
def calc_limits(y):
    # y = y.transpose()[0]
    area = np.trapz(y, list(range(256)), dx=0)
    print("All area =", area)
    print("Area to subtract =", area * percent)
    # indexes of first non-zero elem from start and end
    non_zeros = np.nonzero(y)[0]
    right = non_zeros[-1]
    left = non_zeros[0]
    print("left =", left, "right =", right)
    part = np.trapz(y[left:right + 1], list(range(left, right + 1)), dx=0)
    print("part =", part)
    # while hist area hasn't decreased by 5%
    while area - part < area * percent:
        new_left = left
        new_right = right
        if y[left] < y[right]:
            new_left += 1
        else:
            new_right -= 1
        part = np.trapz(y[new_left:new_right + 1], list(range(new_left, new_right + 1)), dx=0)
        print("left =", new_left, "right =", new_right)
        print("part =", part)
        if area - part > area * percent:
            return new_left, new_right
        else:
            left = new_left
            right = new_right
    return left, right


# calc_hist
def calc_hist(matrix) -> list:
    histogram = {}

    for index in range(0, 256):
        histogram[index] = 0
    for row in matrix:
        for elem in row:
            for item in elem:
                histogram[item] = histogram.get(item, 0) + 1
    return list(histogram.values())


# read image
image = cv2.imread(img_path)

# calculate first hist
hist = calc_hist(image)
plt.plot(hist, color='b')

a, b = calc_limits(hist)
print("a =", a, "b =", b)

# stretch image
rows = len(image)
cols = len(image[0])
print("image: rows =", rows, "cols =", cols)
image2 = deepcopy(image)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
for i in range(rows):
    for j in range(cols):
        _h, _s, _v = image2[i][j]
        _v = stretch(_v)
        image2[i][j] = [_h, _s, _v]
image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)

# calculate second hist
hist2 = calc_hist(image2)
plt.plot(hist2, color='r')

# show hists on plot
plt.show()

# show images
image = np.hstack((image, image2))
cv2.imshow("Linear extension", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
