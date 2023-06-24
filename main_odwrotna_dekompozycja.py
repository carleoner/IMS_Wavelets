import pywt #https://pywavelets.readthedocs.io/en/latest/
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# get image
img = pywt.data.camera() #.aero()
#plt.imsave('./data/cameraorg.png', img.astype('uint8'),cmap = 'gray')

# choose a wavelet
wavelet_name = 'haar'
# calc maxLevel
maxLevel = int(np.floor(np.log2(img.shape[0])))
print('Max decomposition level: '+str(maxLevel))

# greyscale conversion
if len(img.shape) > 2:
    image = np.mean(img, axis=2)

# wavelet decomposition
coeffs = pywt.wavedec2(img, wavelet_name, level=1)
# wavelet reconstruction
reconstructed_image = pywt.waverec2(coeffs, wavelet_name)

# plot
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Oryginalny obraz')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Odtworzony obraz')
plt.axis('off')
plt.show()
