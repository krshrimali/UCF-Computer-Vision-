import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def draw_hist(images):
    img  = images[0]
    img2 = images[1]

    histogram  = []
    histogram2 = []

    rows      = img.shape[0]
    cols      = img.shape[1]
    channels  = img.shape[2]   
    
    rows2     = img2.shape[0]
    cols2     = img2.shape[1]
    channels2 = img2.shape[2]
    if(channels2 != channels):
        print("columns/rows/channels should be same of both for them to be compared.")
        return -1
    for index in range(len(img)):
         for k in range(channels):
            if(index == 0):
                if(k == 0):
                    plt.title("Pixel Intensity Graph")
        
                for i in range(cols):
                    histogram.append(0)
                    for j in range(rows):
                        histogram[i] += img[j][i][k]
                    histogram[i] = histogram[i]/rows
        
                for i in range(cols2):
                    histogram2.append(0)
                    for j in range(rows2):
                        histogram2[i] += img2[j][i][k]
                    histogram2[i] = histogram2[i]/rows2
    
                print("Number of channels: " + str(channels))
    
                if(k == 0):
                    channel = 'Blue'
                    plt.subplot(611)
                    plt.plot(histogram, color='b', label='BLUE CHANNEL')
                    plt.legend(loc="upper right")
        
                elif(k == 1):
                    channel = 'Green'
                    plt.subplot(612)
                    plt.plot(histogram, color='g', label='GREEN CHANNEL')
                    plt.legend(loc="upper right")
    
                elif(k == 2):
                    channel = 'Red'
                    plt.subplot(613)
                    plt.plot(histogram, color='r', label='RED CHANNEL')
                    plt.legend(loc="upper right")
    
            if index == 1:
                if(k == 0):
                    channel = 'Blue'
                    plt.subplot(614)
                    plt.plot(histogram2, color='b', label='BLUE CHANNEL')
                    plt.legend(loc="upper right")
        
                elif(k == 1):
                    channel = 'Green'
                    plt.subplot(615)
                    plt.plot(histogram2, color='g', label='GREEN CHANNEL')
                    plt.legend(loc="upper right")
    
                elif(k == 2):
                    channel = 'Red'
                    plt.subplot(616)
                    plt.plot(histogram2, color='r', label='RED CHANNEL')
                    plt.legend(loc="upper right")
    
                # plt.imshow("Plot")    
                # plt.imshow("Plot")
    plt.show()

def add_noise(img):
    '''
    adds noise randomly
    returns new image

    parameters: image read
    return type: new image (with added noise)
    '''
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
    return img

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

# draw_hist(img)

rows = img.shape[0]
cols = img.shape[1]

img2 = add_noise(img)

plt.imshow(img2)
plt.show()

draw_hist([img, img2])
