import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import random

img = cv2.imread(sys.argv[1])

histogram = []

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
channels = img.shape[2]

# print(img[0][10])
# considering only 1 channel

plt.imshow(img)
plt.show()

# resize image to 50% down (resolution) to perform operations
# reduces cost and time complexity

img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
rows = img.shape[0]
cols = img.shape[1]

for k in range(channels):
    for i in range(rows):
        for j in range(cols):
            inc = random.randint(0, 100)
            if(img[i][j][k] + inc < 255):
                img[i][j][k] += inc
            else:
                img[i][j][k] -= inc

img = cv2.resize(img, (0,0), fx = 2.0, fy = 2.0)
rows = img.shape[0]
cols = img.shape[1]

plt.imshow(img)
plt.show()

for k in range(channels):
    if(k == 0):
        plt.title("Pixel Intensity Graph")

    for i in range(cols):
        histogram.append(0)
        for j in range(rows):
            histogram[i] += img[j][i][k]
    
    histogram[i] = histogram[i]/rows
    
    if(k == 0):
        channel = 'Blue'
        plt.subplot(311)
        plt.plot(histogram, color='b', label='BLUE CHANNEL')
        plt.legend(loc="upper right")

    elif(k == 1):
        channel = 'Green'
        plt.subplot(312)
        plt.plot(histogram, color='g', label='GREEN CHANNEL')
        plt.legend(loc="upper right")

    elif(k == 2):
        channel = 'Red'
        plt.subplot(313)
        plt.plot(histogram, color='r', label='RED CHANNEL')
        plt.legend(loc="upper right")

    # plt.imshow("Plot")
plt.show()

