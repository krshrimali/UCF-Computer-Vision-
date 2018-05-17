import cv2
import numpy as np
import utils as util
import sys

def derivative(img, axis=0):
    # sobel x derivative if axis = 0, else y derivative (if axis = 1)
    img_new = np.diff(image, axis=axis) # differentiates image along the axis
    util.show(img_new, 0) # display the image onto the screen

def help():
    print("--------------------------------------------------")
    print("Arguments: ")
    print("Example usage 1 : python3 edge_filter.py <image_path> 0")
    print("Example usage 2 : python3 edge_filter.py <image_path>") 

    print("First argument:  (compulsory) " + str("image path"))
    print("Second argument: (optional)   " + str("axis "))
    print("--------------------------------------------------")

if __main__ == "__main__":
    # if arguments not given
    if(len(sys.argv) == 1):
        help()
        print("Exiting...")
        sys.exit()
    # read image
    image = cv2.imread(sys.argv[1], 0)
    util.show(image, 0)
    
    if(len(sys.argv) == 2):
        derivative(image, sys.argv[2])
    else:
        derivative(image)

main()
