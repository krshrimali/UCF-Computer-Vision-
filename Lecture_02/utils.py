import cv2

def show(img, path=1):
    '''
    function to show image read by the user
    parameter: image path or image read and path variable
    path = 1 if path given
    path = 0 if read image given
    returns image read by the user
    '''
    if(path):
        img2 = cv2.imread(img)
        cv2.imshow(str(img), img2)
    else:
        cv2.imshow("Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if(path):
        return img2
    else:
        return img


def draw_hist(imgPath, img, full = None):
    print("Drawing Histogram")
    print("You want histograms of R, G, B Channels in one image or separate?")
    decide = int(input("separate or single? 0/1 "))
    flag = 0
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], full, [256], [0,256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])
        if(decide == 0):
            plt.show()
            plt.savefig(imgPath + '_' + col + '.png')
            flag += 1
            print(flag)
    if(flag != 3):
        plt.show()
        plt.savefig(imgPath + '.png')

