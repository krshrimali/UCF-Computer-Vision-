# gaussian smoothing filter
    # g(x) = 1/(root(2 * pi) * sigma) * exp(-(x2 + y2)/(2 * sigma2))
# task is to apply this filter, x is pixel value

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def gaussian_filter(input_img, sigma):
    (rows, cols) = input_img.shape
    print("Rows: " + str(rows) + ", Cols: " + str(cols))
    g = []
    
    for i in range(rows):
        g.append([])
        for j in range(cols):
            g[i].extend([int(255 * np.exp(-(input_img[i][j]**2)/(2 * sigma**2)))]) 
    return g

img  = cv2.imread("/home/krshrimali/Pictures/Kush.jpg", 0)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
filtered = gaussian_filter(img, 2)
print(filtered)
print("Image: ")
for i in range(100):
    print("")
print(img)
print(len(filtered[0]))
print(img.shape)

np.asarray(filtered)


plt.imshow(filtered, interpolation='nearest', cmap = plt.get_cmap('gray'))
plt.show()
