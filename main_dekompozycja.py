import pywt #https://pywavelets.readthedocs.io/en/latest/
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

wavelet_name = pywt.Wavelet('haar')
print(wavelet_name) #print out wavelet properties
print(pywt.families(short=False)) #print out available wavelet families
for family in pywt.families(): #print out available wavelet names in each family
    print("%s family: " %family + ', '.join(pywt.wavelist(family)))

# Lenna
# img_url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
# img = Image.open(requests.get(img_url, stream=True).raw) #img = mpimg.imread('./Lenna.png')
# img = img[:, :, 2] # wybranie kanalu - w zalenosci od przestrzeni barw (RGB - 2nieb, 1zielony, 0czerwony) #np.mean(img,-1)
# print(img.shape)

# Camera lub aero
img = pywt.data.camera() #.aero()

# imgplot = plt.imshow(img)
# plt.show()

# calc maxLevel
maxLevel = int(np.floor(np.log2(img.shape[0])))

############################
############################

coeffs = pywt.dwt2(img, wavelet_name, mode='periodization') # dwt - decomposition, idwt - deconstruction
AA, (HD, VD, DD) = coeffs
fig = plt.figure(figsize=(10, 10))

titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
for i, co in enumerate([AA, HD, VD, DD]):
    subplt = fig.add_subplot(2, 2, i + 1)
    subplt.imshow(co, cmap= 'gray')
    subplt.set_title(titles[i], fontsize=10)
    subplt.axis('off')

#plt.imsave('./data/name'+'.png',img,cmap = 'gray')
plt.show()

############################
############################

coeffs = pywt.wavedec2(img, wavelet_name, mode='periodization', level=2) # dwt - decomposition, idwt - deconstruction
AA, (HD, VD, DD), (HDD, VDD, DDD) = coeffs
fig = plt.figure(figsize=(10, 10))

titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail', 'Horizontal detail',
          'Vertical detail', 'Diagonal detail']
for i, co in enumerate([AA, HD, VD, DD, HDD, VDD, DDD]):
    subplt = fig.add_subplot(1, 7, i + 1)
    subplt.imshow(co) #, cmap= 'gray')
    subplt.set_title(titles[i], fontsize=10)
    subplt.axis('off')

# #plt.imsave('./data/name'+'.png',img,cmap = 'gray')

# # plt.subplots_adjust(left=0,
# #                     bottom=0,
# #                     right=0.806,
# #                     top=1,
# #                     wspace=0,
# #                     hspace=0)

# # plt.subplot_tool()
plt.show()


##########################
##########################

coeffs = pywt.wavedec2(img,wavelet_name,level = 2,mode = 'periodization')

# normalize coeffs
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(2):
    print(detail_level)
    
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]]


arr, coeff_slices = pywt.coeffs_to_array(coeffs)

plt.imshow(arr,cmap='gray')
plt.axis('off')
plt.title('level = '+str(2))
plt.show()
