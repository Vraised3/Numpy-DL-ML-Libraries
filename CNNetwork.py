import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage import transform # To change size

def convolve2d(img, filt, pad = 0, stride = 1):
    rows, cols = img.shape
    print(rows, cols)
##    assert(rows!=cols, 'image not of equal dimensions') # not always possible
    fis = round((rows + 2*pad - 3) / stride) + 1
    print(fis)
    newMatrix = np.zeros((fis, fis))
    for i in range(pad + 1, fis):
        for j in range(pad + 1, fis):
            sub = img[i - 1:i + 2, j - 1:j + 2] #3x3 filter
            newMatrix[i - 1][j - 1] = np.sum(filt * sub)
    print(img.shape)
    print(newMatrix.shape)
    return newMatrix#[1:-1, 1:-1]
# Filtered image size = (n+2p-f)/s + 1
# n = image size
# p = padding
# f = filter size
# s = stride

def convolve3d(img, filt, appl = [1, 1, 1]):
    conv_H = convolve2d(img[:, :, 0], filt) * appl[0]
    conv_S = convolve2d(img[:, :, 1], filt) * appl[1]
    conv_V = convolve2d(img[:, :, 2], filt) * appl[2]
    return conv_H + conv_S + conv_V #np.dstack((conv_H, conv_S, conv_V))

        
##img = np.random.rand(50,50)

img = misc.face()
img = transform.resize(img, (255,255))
plt.imshow(convolve3d(img,np.array([[1,0,-1],[1,0,-1],[1,0,-1]])))
plt.show()

img_gray = np.dot(img, [0.299, 0.587, 0.114])

conv_fl = np.array([[[1,0,1],[0,1,0],[1,0,1]],
           [[0,0,1],[0,1,0],[1,0,0]],
           [[1,0,-1],[1,0,-1],[1,0,-1]]])
for kernel in conv_fl:
##    conv_fl_arr = np.array(kernel)
    print(kernel)
    img_conv = convolve2d(img_gray, kernel)
    plt.figure()
    plt.imshow(img_conv, cmap='gray')
    plt.show()
