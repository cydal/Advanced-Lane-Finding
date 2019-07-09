import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.signal import find_peaks_cwt
import os


distortion = pickle.load( open( "distort.p", "rb" ) )
mtx, dist = distortion["mtx"], distortion["dist"]

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    x, y = None, None
    
    if orient == "x":
        x, y = 1, 0
    else:
        x, y = 0, 1
        
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, x, y)
    
    sobelxy = np.absolute(sobelxy)
    
    scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
    
    sxbinary = np.zeros_like(scaled_sobel)
    
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    grad_binary = sxbinary
    
    return grad_binary


def sobel_combine(img):
# Choose a Sobel kernel size
    ksize = 5# Choose a larger odd number to smooth gradient measurements

    blurImg = img
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(blurImg, orient='x', thresh=(25, 205))
    grady = abs_sobel_thresh(blurImg, orient='y', thresh=(25, 205))
    binary = cv2.bitwise_and(gradx, grady)
    return binary



def combined_threshold(img, ksize=11):
    color_binary = LandS(img)
    sobel_binary = sobel_combine(img)
    b = bMask(img)
    r = rMask(img)
    u = uMask(img)
    y = yMask(img)
    v = van(img)
    #dw = darkWhite(img)
    dy = darkYellow(img)
    binary = np.zeros_like(sobel_binary)
    binary[(color_binary == 1) | (sobel_binary == 1) | ((b == 1) & (r == 1)) | (u == 1) | (y == 1) | (v == 1) | (dy == 1)] = 1
    #binary = gaussian_blur(binary, kernel=ksize)
    return binary





def hlsMask(img):

    S = img[:, :, 2]

    thresh = (150, 255)
    Sbinary = np.zeros_like(S)
    Sbinary[(S >= thresh[0]) & (S <= thresh[1])] = 1
    return Sbinary

def lMask(img):

    L = img[:, :, 1]

    thresh = (150, 255)
    Lbinary = np.zeros_like(L)
    Lbinary[(L > thresh[0]) & (L <= thresh[1])] = 1
    return Lbinary




def rMask(img):

    R = img[:, :, 0]

    thresh = (220, 255)
    Rbinary = np.zeros_like(R)
    Rbinary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return Rbinary

def bMask(img):

    R = img[:, :, 2]

    thresh = (180, 255)
    Rbinary = np.zeros_like(R)
    Rbinary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return Rbinary


def yMask(img):

    Y = img[:, :, 0]

    thresh = (230, 250)
    ybinary = np.zeros_like(Y)
    ybinary[(Y > thresh[0]) & (Y <= thresh[1])] = 1
    return ybinary

def uMask(img):

    Y = img[:, :, 1]

    thresh = (200, 250)
    ybinary = np.zeros_like(Y)
    ybinary[(Y > thresh[0]) & (Y <= thresh[1])] = 1
    return ybinary


def van(img):
    rthresh = [170, 230]
    gthresh = [190, 220]
    bthresh = [190, 240]
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    binary = np.zeros_like(R)
    rbinary = np.zeros_like(R)
    gbinary = np.zeros_like(R)
    bbinary = np.zeros_like(R)
    rbinary[(R <= rthresh[1]) & (R >= rthresh[0])] = 1
    gbinary[(G <= gthresh[1]) & (G >= gthresh[0])] = 1
    bbinary[(B <= bthresh[1]) & (B >= bthresh[0])] = 1
    binary[(rbinary == 1) & (gbinary == 1) & (bbinary == 1)] = 1

    return binary


def darkYellow(img):
    rthresh = [150, 230]
    gthresh = [130, 170]
    bthresh = [40, 70]
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    binary = np.zeros_like(R)
    rbinary = np.zeros_like(R)
    gbinary = np.zeros_like(R)
    bbinary = np.zeros_like(R)
    rbinary[(R <= rthresh[1]) & (R >= rthresh[0])] = 1
    gbinary[(G <= gthresh[1]) & (G >= gthresh[0])] = 1
    bbinary[(B <= bthresh[1]) & (B >= bthresh[0])] = 1
    binary[(rbinary == 1) & (gbinary == 1) & (bbinary == 1)] = 1
    return binary


def LandS(img):
    
    imageHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    binary = np.zeros((img.shape))
    binary = binary.mean(2)
    G = hlsMask(imageHLS)
    R = lMask(imageHLS)
    binary[(G == 1) & (R == 1)] = 1
    return binary



def gaussian_blur(img, kernel=33):
    blur = cv2.GaussianBlur(img,(kernel,kernel),0)
    return blur


def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)


src = np.float32([[543,458],[720,460],[1233,710],[193,712]])

dst = np.float32([[100,100],[1100,100],[1100,720],[100,720]])

src = np.float32([[570,458],[720,460],[1233,710],[210,712]])

dst = np.float32([[100,100],[1100,100],[1100,720],[100,720]])


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def preprocess(image):
    image = combined_threshold(image)
    image = warp(image)
    #image[int(image.shape[0]/2):, :] = gaussian_blur(image[int(image.shape[0]/2):, :])
    binary = gaussian_blur(image, kernel=21)
    return binary


def inv():
    return cv2.getPerspectiveTransform(dst, src)


def cent(center):
    
    return "{}{}{}".format("Left of Center: ", abs(center), "m") if center < 0 else "{}{}{}".format("Right of Center: ", abs(center), "m")


