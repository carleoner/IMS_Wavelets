import pywt #https://pywavelets.readthedocs.io/en/latest/
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

# get image
img = pywt.data.camera() #.aero()

# choose a wavelet
wavelet_name = 'haar'

# calc maxLevel
maxLevel = int(np.floor(np.log2(img.shape[0])))
print(maxLevel)

mean = 0  # Średnia wartość szumu
std_dev = 0.4  # Odchylenie standardowe szumu
noise = np.random.normal(mean, std_dev, img.shape).astype(np.uint8)  # Generuj szum gaussowski
#noisy_img = cv2.add(img, noise)  # Dodaj szum do obrazu
noisy_img = random_noise(img, mode='speckle')

plt.imshow(noisy_img, cmap=plt.get_cmap("gray"))
plt.axis('off')
plt.show()

# calc sigma
sigma_est = estimate_sigma(noisy_img, average_sigmas=True)
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

img_denoise = denoise_wavelet(noisy_img, method='VisuShrink', mode='soft', wavelet=wavelet_name, wavelet_levels=9, sigma=sigma_est/3, rescale_sigma=True)

psnr = peak_signal_noise_ratio(img, img_denoise)
print(psnr)

plt.imshow(img_denoise, cmap=plt.get_cmap("gray"))
plt.axis('off')
plt.show()
