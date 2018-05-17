'''
@author: Kushashwa Ravi Shrimali
UCF Computer Vision Course + Computer Vision Algorithms and Applications, Richard Szeliski 
Next TO-DO: Compositing Equation : C = (1 - alpha)*B + alpha*F, B is background and Foreground F
'''

import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import math as m
import utils

def cumsum(h):
    # credits:://gist.github.com/bistaumanga/6309599 
    list_ = []
    for i in range(len(h)):
        list_.append(sum(h[:i+1]))
    
    print(i, len(h))
    print("FINISHED")
    # return [sum(h[:i+1] for i in range(len(h)))]
    return list_

def imhist(img):
    # normalized histogram of image
    # credits:://gist.github.com/bistaumanga/6309599 

    m, n = img.shape
    h    = [0.0] * 256 # create 256 length array, zero elements
    
    for i in range(m):
        for j in range(n):
            h[img[i, j]] += 1
    
    return np.array(h)/(m*n)

def equalize_hist(img):
    img         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape[0], img.shape[1])
    # credits:://gist.github.com/bistaumanga/6309599 
    h           = imhist(img)
    cdf         = np.array(cumsum(h))
    sk          = np.uint8(255 * cdf)
    s1, s2      = img.shape
    Y           = np.zeros_like(img)
    # applying transferred values for each pixels
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[img[i, j]]

    H = imhist(Y)

    # return transformed image, original new histogram, transform function
    return Y, h, H, sk

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
                    plt.subplot(131)
                    plt.plot(histogram, color='b', label='B')
                    plt.legend(loc="upper right")
        
                elif(k == 1):
                    channel = 'Green'
                    plt.subplot(132)
                    plt.plot(histogram, color='g', label='G')
                    plt.legend(loc="upper right")
    
                elif(k == 2):
                    channel = 'Red'
                    plt.subplot(133)
                    plt.plot(histogram, color='r', label='R')
                    plt.legend(loc="upper right")
    
            if index == 1:
                if(k == 0):
                    channel = 'Blue'
                    plt.subplot(131)
                    plt.plot(histogram2, color = 'b', ls = '--', label='B(Noisy)')
                    plt.legend(loc="upper right")
        
                elif(k == 1):
                    channel = 'Green'
                    plt.subplot(132)
                    plt.plot(histogram2, color = 'g', ls = '--', label='G(Noisy)')
                    plt.legend(loc="upper right")
    
                elif(k == 2):
                    channel = 'Red'
                    plt.subplot(133)
                    plt.plot(histogram2, color = 'r', ls = '--', label='R(Noisy)')
                    plt.legend(loc="upper right")
    
                # plt.imshow("Plot")    
                # plt.imshow("Plot")
    plt.show()

def gamma_correction(img):
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    rows = img.shape[0]
    cols = img.shape[1]

    for k in range(channels):
        for i in range(rows):
            for j in range(cols):
                img[i][j][k] = m.pow(img[i][j][k], 1.0/2.2)

    img = cv2.resize(img, (0,0), fx = 2.0, fy = 2.0)
    return img


def add_noise(img):
    '''
    adds noise randomly
    returns new image

    parameters: image read
    return type: new image (with added noise)
    '''
    img      = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    rows     = img.shape[0]
    cols     = img.shape[1]
    channels = img.shape[2]

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

def over_operator():
    '''
    Porter and Duff (1984)
    Blinn (1994a, 1994b)
    '''
    default_path = "/home/krshrimali/Pictures/"

    B                         = default_path + str(input("Background image path: "))
    F                         = default_path + str(input("Foreground image path: "))
    transparency_factor_alpha = float(input("Transparency Factor alpha: "))
    '''
    B = "/home/krshrimali/Pictures/background.jpg"
    F = "/home/krshrimali/Pictures/Kush.jpg"
    transparency_factor_alpha = float(0.5)
    '''

    B_img                     = cv2.imread(B)
    F_img                     = cv2.imread(F)
    
    B_img                     = cv2.resize(B_img, (400, 400), interpolation = cv2.INTER_AREA)
    F_img                     = cv2.resize(F_img, (400, 400), interpolation = cv2.INTER_AREA)
    C                         = cv2.addWeighted(B_img, 1 - transparency_factor_alpha,
            F_img, transparency_factor_alpha, 0)

    # C = (1 - transparency_factor_alpha) * B_img + transparency_factor_alpha * F_img
    
    utils.show(B)
    utils.show(F)
    utils.show(C,0)

def main():
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
    
    # img2 = gamma_correction(img)
    
    noise_or_not = input("You want to add noise to the image?")
    if(noise_or_not.lower() == "yes"):
        img2 = add_noise(img)
        plt.imshow(img2)
        plt.show()
        draw_hist([img, img2])
    else:
        imgPath = input("give argument to the second image: ")
        img2 = cv2.imread(imgPath)
        plt.imshow(img2)
        plt.show()
        draw_hist([img, img2])
    
    over_operator()
    '''
    print("Now applying cdf to the original image, histogram equalization ")
    new_img, original_hist, new_hist, transform_func = equalize_hist(img)
        
    cv2.imshow("new_img", new_img)
    cv2.imwrite("new_img.png", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.plot(original_hist, color = 'b', ls = '--', label = 'Original Histogram')
    plt.legend(loc="upper right")
    plt.plot(new_hist, color = 'r', label = 'New Histogram')
    plt.legend(loc="upper right")
    
    plt.show()
    '''
    '''
    plt.imshow(img2)
    plt.show()
    draw_hist([img, img2])
    '''
main()
