import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import sys


# Load images
"""
Images fetched from links provided in task:
% https://pixabay.com/en/board-chess-chessboard-black-white-157165/
% https://pixabay.com/en/jellyfish-under-water-sea-ocean-698521/
% https://pixabay.com/en/new-york-city-skyline-nyc-690868/
"""

def svd_compression_r(image, r):
    """
    :param image: input image in grayscale (vals between
                    0-255)
    :param r: Number of singular values to keep.
    :return: singular values, compressed image
    """

    # Scale image to have values between 0 and 1
    scaled_image = image/255

    # Singular Value Decomposition
    u, s, vh = np.linalg.svd(scaled_image)

    # Truncate matrices to keep only r first values
    # and reconstruct to image.
    compressed_image = u[:, :r].dot(np.diag(s[:r]).dot(vh[:r, :]))
    return compressed_image, s

def get_image_size(image):
    """
    Return amount of memory needed to store an image. 
    Size is returned in bytes.
    """
    return np.size(image)*np.itemsize(image)


im1 = imread('images/chessboard.png', as_gray=True)
im2 = imread('images/jellyfish.jpg', as_gray=True)
im3 = imread('images/new_york.jpg', as_gray=True)

imgs = [im1, im2, im3]
r = int(sys.argv[1])
comps = []
s_vals = []
for img in imgs:
    comp, s = svd_compression_r(img, r)
    comps.append(comp)
    s_vals.append(s)

# Plotting
## Plot log of singular values
plt.plot(np.log(s_vals[0]), 'r', label=r'chessboard')
plt.plot(np.log(s_vals[1]), 'g', label=r'jellyfish')
plt.plot(np.log(s_vals[2]), 'b', label=r'new york')
plt.xlabel(r'#$\sigma$ descending order')
plt.ylabel(r'$log(\sigma)$')
plt.legend()
plt.show()

## Plot image and compressed image side by side
for comp, img in zip(comps, imgs):
    plt.subplot(1,2,1)
    plt.gca().set_title('Compressed')
    plt.imshow(comp, cmap='gray')
    plt.subplot(1,2,2)
    plt.gca().set_title('Original')
    plt.imshow(img, cmap='gray')
    plt.show()

print("Comparison of image sizes")
print("Original (MB) | Compressed (MB)")
for comp, img in zip(comps, imgs):
    print

