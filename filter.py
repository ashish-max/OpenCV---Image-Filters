import cv2
import numpy as np

from scipy.interpolate import UnivariateSpline

img = cv2.imread("cat.png")
print('---ORIGINAL---')


#IMG RESIZING
scale_percent = 0.60
width = int(img.shape[1]*scale_percent)
height = int(img.shape[0]*scale_percent)

dim = (width,height)
resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
print('1' + ' Image Resized.')


#SHARPENING EFFECT
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
sharpened = cv2.filter2D(resized,-1,kernel_sharpening)
print('2' + ' Image Sharpened.')


#GRAY SCALE
gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
print('3' + ' Image Grayscaled.')


#INVERSE EFFECT
inv = 255-gray
print('4' + ' Negative Image created.')


#GUSSIAN BLUR EFFECT
gauss = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)
print('5' + ' Gaussian Blur Applied.')


#PENCIL SKETCH EFFECT
def dodgeV2(image,mask):
    return cv2.divide(image,255-mask,scale=256)
pencil_img = dodgeV2(gray,gauss)
print('6' + ' Pencil Sketch effect Applied.')


#Cooling effect
class CoolingFilter():

    def __init__(self):
        #Initialize look-up table for curve filter
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        # warming filter: increase red, decrease blue
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))


        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        #Creates a look-up table using scipy's spline interpolation
        spl = UnivariateSpline(x, y)
        return spl(range(256))
print('7' + ' Cooling Effect Applied.')
    
#Warming effect
class WarmingFilter():

    def __init__(self):
        #Initialize look-up table for curve filter
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        # cooling filter: increase blue, decrease red
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        #Creates a look-up table using scipy's spline interpolation
        spl = UnivariateSpline(x, y)
        return spl(range(256))
print('8' + ' Warming  Effect Applied.')    
#creating class objects to access effects
x = WarmingFilter()
Warm  = x.render(resized)

y = CoolingFilter()
Cool = y.render(resized)


#showing Effects on imgs
cv2.imshow('Original',img)
cv2.imshow('resized',resized)
cv2.imshow('sharp',sharpened)
cv2.imshow('gray',gray)
cv2.imshow('inv',inv)
cv2.imshow('gauss',gauss)
cv2.imshow('pencil sketch',pencil_img)
cv2.imshow('Warm filter',Warm)
cv2.imshow('Cool filter',Cool)

cv2.waitKey(0)
cv2.destroyAllWindows()
