from edge_filter import derivative
import cv2, sys, utils
from scipy import signal # for convolving images
import numpy as np

# derivative(img, axis) returns derivative of the image along the axis
# aim : if(derivative(img, axis = 0) = (I conv [[-1 0 -1] [2 0 -2] [1 0 -1]] )
# = True

def just_func():
    print("just func")

# def conv(mat1, mat2):
if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0) # read the image 
    utils.show(img, 0)
    image_new = derivative(img, 0) # derivative of the image along x axis
    mask   = np.array([[-1, 0, -1], [2, 0, -2], [1, 0, -1]]) # mask for convolving
    mask_without_smoothing = np.array([1, 0, -1])
    # fx_img = conv(img, mask) # returns convolved output 
    fx_img = cv2.filter2D(img, -1, mask) # same for zero padding
    fx_img_original = cv2.filter2D(img, -1, mask_without_smoothing)
    utils.show(fx_img_original, 0)

