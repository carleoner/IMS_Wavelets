import pywt #https://pywavelets.readthedocs.io/en/latest/
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib

# get image
img = pywt.data.camera()
print(img)
# plt.show()

# choose a wavelet
wavelet_name = 'db1'
# calc maxLevel
maxLevel = int(np.floor(np.log2(img.shape[0])))

############################
############################

# perform wavelet transform
coeffs = pywt.wavedec2(img, wavelet = wavelet_name, level = maxLevel)
# convert coefficients to an array 
coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
coeff_arr_reshaped = coeff_arr.reshape(-1)
# the array is sorted
coeffs_sorted = np.sort(np.abs(coeff_arr_reshaped))

vals = [1, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
file_sizes = []

for value in vals:
    # create a treshhold
    threshholdVal = int(np.floor((1-value)*len(coeffs_sorted)))
    threshholdIndex = coeffs_sorted[threshholdVal]
    # create a mask: true or false if val > threshholdIndex
    ind = np.abs(coeff_arr) > threshholdIndex
    coeff_arr_filtered = coeff_arr * ind
    # convert a combined array of coeffs back to a list compatible with waverec2
    coeffs_filtered = pywt.array_to_coeffs(coeff_arr_filtered, coeff_slices, output_format='wavedec2')
    
    compressed_img = pywt.waverec2(coeffs_filtered, wavelet=wavelet_name).astype('uint8')
    #print(compressed_img - img)
    plt.figure()
    plt.axis('off')
    #compressed_img -= img
    plt.imshow(compressed_img,cmap='gray')
    plt.show()
    plt.imsave('./data/c'+str(value)+'.png', compressed_img.astype('uint8'),cmap = 'gray')
    path = pathlib.Path('./data/c'+str(value)+'.png')
    print(str(value) + '. ' + str(len(coeffs_sorted)) + ' ' + str(path.stat().st_size) + ' Bytes')
    # print(str(path.stat().st_size) + ' Bytes, ' + str(round(path.stat().st_size * 100 / pathlib.Path('./data/original.png').stat().st_size)) + '%')
    file_sizes.append(path.stat().st_size)

############################
############################

plt.figure()
plt.plot(vals, file_sizes, 'ro')
plt.grid('on')
plt.axis([0, 1, 0, None])
plt.xlabel('%')
plt.ylabel('file size [Bytes]')
plt.show()
plt.savefig('./data/wykres.png')
