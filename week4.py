# import cv2
# import imageio
# import matplotlib.pyplot as plt
# from skimage.viewer import ImageViewer
# from skimage.color import rgb2hsv
# from skimage.color import hsv2rgb
#
# im = cv2.imread("banaan.jpg")
# im2 = cv2.imread("banaan_nieuw.png")
#
#
# # for i in im:
# #     for j in i:
# #         if(j[0] < 245 and j[1] < 245):
# #             j[0] *= 0.15
# #             j[1] *= 0.15
# #             j[2] *= 0.15
# #             j[0] += 30
# #             j[1] += 30
# #             j[2] += 30
#
# im3 = rgb2hsv(im)
# colors = im3[:, : , 0].flatten()
# plt.hist(colors)
# # plt.hist(im2[0])
# #
# plt.show()
# #
# # im4 = hsv2rgb(im3)
# print(colors)
# # viewer = ImageViewer(im)
# # viewer.show()


#week 2
# import skimage.filters.edges
# from skimage import data, filters, feature
# from skimage.util import random_noise
# from skimage.viewer import ImageViewer
# import scipy
# import matplotlib.pyplot as plt
#
#
# fig, ax = plt.subplots(2, 4, figsize=(10,5))
#
# image = data.camera()
# ax[0,0].imshow(image, cmap='gray')
# mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
# newimage=scipy.ndimage.convolve(image, mask1)
# newimage=scipy.ndimage.convolve(newimage, mask1)
#
# ima = newimage.copy()
# ax[0,1].imshow(newimage, cmap='gray')
#
# newimage = ima
# mask1=[[-1,0,1],[-1,0,1],[-1,0,1]]
# newimage=scipy.ndimage.convolve(newimage, mask1)
# newimage=scipy.ndimage.convolve(newimage, mask1)
# ax[0,2].imshow(newimage, cmap='gray')
#
# mask1=[[-1,-1,-1],[0,0,0],[1,1,1]]
# newimage = ima
# newimage=scipy.ndimage.convolve(newimage, mask1)
# newimage=scipy.ndimage.convolve(newimage, mask1)
# ax[0,3].imshow(newimage, cmap='gray')
#
# mask1=[[1,2,1],[0,0,0],[-1,-2,-1]]
# newimage = ima
# newimage=scipy.ndimage.convolve(newimage, mask1)
# newimage=scipy.ndimage.convolve(newimage, mask1)
# ax[1,0].imshow(newimage, cmap='gray')
#
#
# newimage = ima
#
# newimage = skimage.filters.edges.farid(newimage)
# ax[1,1].imshow(newimage, cmap='gray')
#
# newimage = ima
#
# newimage = skimage.filters.edges.laplace(newimage)
# # ax[1,2].imshow(newimage, cmap='gray')
#
# newimage = ima
#
# newimage = skimage.filters.edges.roberts(newimage)
# # ax[1,3].imshow(newimage, cmap='gray')
#
# #Het valt me op dat de ingebouwde edge functies veel duidelijkere edges terug geven met veel minder ruis,
# #Het valt me ook op dat het resultaat van 3 verschillende gebruikte functies gelijk aan elkaar is.
#
# newimage = ima
# from scipy import ndimage as ndi
# # Generate noisy image of a square
# newimage = ndi.gaussian_filter(newimage, 4)
# newimage = random_noise(newimage, mode='speckle', mean=0.1)
#
# # Compute the Canny filter for two values of sigma
# edges2 = feature.canny(newimage, sigma=3)
# edges1 = feature.canny(newimage, sigma=5)
#
#
# ax[1,2].imshow(edges2, cmap='gray')
# ax[1,3].imshow(edges1, cmap='gray')
#
#
# fig.tight_layout()
# plt.show()

#Week 4
import random
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

data = []
target = []
images = []

numbers = []

for i in range(len(digits.data)):
    numbers.append(i)


def Testdigits():
    j = random.randint(0, len(digits.data))
    if j in numbers:
        data.append(digits.data[j])
        target.append(digits.target[j])
        images.append(digits.images[j])
        numbers.remove(j)
    else:
        Testdigits()

for i in range(int(len(digits.data)*0.3333)):
    Testdigits()

print("aantal leer digits: " + str(len(numbers)))

clf = svm.SVC(gamma=0.001, C=100)
X = []
y = []
for i in numbers:
    X.append(digits.data[i])
    y.append(digits.target[i])

clf.fit(X,y)


testdigits = [data,target]

result = 0
predictions = clf.predict(testdigits[0])
for i in range(len(predictions)):
    if target[i] == predictions[i]:
        result += 1
print(float(result / len(predictions)))

#Week 5

