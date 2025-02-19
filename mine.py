#from scipy import misc
#from PIL import Image
#from skimage import exposure
#from sklearn import svm
#import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
#import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
#import pandas as pd
import pywt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
immatrix=[]
im_unpre = []
path ="D:\sample\*.*"

for file in glob.glob(path):
    #print(file)
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Color image', img)
    #cv2.imshow('gray image',img_gray)
    equ = cv2.equalizeHist(img_gray) 
    #cv2.imshow('gray image',img_gray)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    immatrix.append(np.array(equ).flatten())

np.shape(immatrix)
print(np.shape(equ))
plt.imshow(immatrix[0].reshape((1000,1000)),cmap='gray')
plt.show()


imm_dwt = []
for equ in immatrix:
    equ = equ.reshape((1000,1000))
    coeffs = pywt.dwt2(equ, 'haar')
    equ2 = pywt.idwt2(coeffs, 'haar')
    imm_dwt.append(np.array(equ2).flatten())
    
np.shape(imm_dwt)
np.shape(equ2)
plt.imshow(imm_dwt[0].reshape((1000,1000)),cmap='gray')
plt.show()    

def _filter_kernel_mf_fdog(L, sigma, t = 3, mf = True):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')
    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2)
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()
        
    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
        #print("moh")
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2
    
    def k_fun(x):
        return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)
    
    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)
    
    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)
    
    return cv2.flip(kernel, -1) 

def show_images(images,titles=None, scale=1.3):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.imshow(image, cmap = cm.Greys_r)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * n_ims / scale)
    plt.show()

def gaussian_matched_filter_kernel(L, sigma, t = 3):
    
   # K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
        
    return _filter_kernel_mf_fdog(L, sigma, t, True)

def createMatchedFilterBank(K, n = 12):
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]
    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)
    return kernels

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)


gf = gaussian_matched_filter_kernel(20, 5)
bank_gf = createMatchedFilterBank(gf, 4)
imm_gauss = []
for equ2 in imm_dwt:
    equ2 = equ2.reshape((1000,1000))
    equ3 = applyFilters(equ2,bank_gf)
    imm_gauss.append(np.array(equ3).flatten())
#print("done")
np.shape(imm_gauss)
plt.imshow(imm_gauss[0].reshape((1000,1000)),cmap='gray')
plt.show()

def createMatchedFilterBank():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

bank_gf = createMatchedFilterBank()
equx=equ3
equ3 = applyFilters(equ2,bank_gf)
imm_gauss2 = []

for equ2 in imm_dwt:
    equ2 = equ2.reshape((1000,1000))
    equ3 = applyFilters(equ2,bank_gf)
    imm_gauss2.append(np.array(equ3).flatten())
np.shape(imm_gauss2)
plt.imshow(imm_gauss2[1].reshape((1000,1000)),cmap='gray')
plt.show()
np.shape(imm_gauss2)
plt.imshow(imm_gauss2[1].reshape((1000,1000)),cmap='gray')
plt.show()


e_ = equ3
np.shape(e_)
e_=e_.reshape((1000,1000))
np.shape(e_)



img = equ3
Z = img.reshape((1000,1000))
Z = np.float32(Z)
k=cv2.KMEANS_PP_CENTERS
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
#cv2.imshow('res2',res2)
imm_kmean = []

for equ3 in imm_gauss2:
    img = equ3.reshape((1000,1000))
    Z = img.reshape((1000,1000))
    Z = np.float32(Z)
    k=cv2.KMEANS_PP_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    imm_kmean.append(np.array(res2).flatten())
    
np.shape(imm_kmean)
plt.imshow(imm_kmean[0].reshape((1000,1000)),cmap="gray")
plt.show()


from sklearn.svm import SVC
clf = SVC()
Y = np.ones(95)
Y[1]=Y[5]=Y[7]=Y[17]=Y[6]=0

clf.fit(imm_kmean, Y)

y_pred = clf.predict(imm_kmean)
k = [1,3,4,9,10,11,13,14,20,22,24,25,26,27,28,29,35,36,38,42]
k = k-np.ones(len(k))
k =[int(x) for x in k]
imm_train = []
y_train = []
k.append(5)
k.append(7)
for i in k:
    imm_train.append(imm_kmean[i])
    y_train.append(Y[i])
    
y_train
clf.fit(imm_train, y_train)
y_pred = clf.predict(imm_kmean)
accuracy_score(Y,y_pred)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(imm_train, y_train) 
y_pred2=neigh.predict(imm_kmean)
neigh.score(imm_kmean,Y)
