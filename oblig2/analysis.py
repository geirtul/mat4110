import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import sys
import os


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

def get_compression_ratio(filename1, filename2):
    """
    Get size of image after it has been saved.
    Compression ratio assumes file 1 is non-compressed
    and file 2 is compressed.
    """

    size1 = os.path.getsize(filename1)
    size2 = os.path.getsize(filename2)
    return size1/size2

# input values
# >python analysis.py <file_to_compress> <r>
filename = sys.argv[1]
r = int(sys.argv[2])
if len(sys.argv) > 3:
    nm = int(sys.argv[3])

# Plot log(sigma) for all images together.
if filename == "sigmaplot":
    im1 = imread('images/chessboard.png', as_gray=True)
    im2 = imread('images/jellyfish.jpg', as_gray=True)
    im3 = imread('images/new_york.jpg', as_gray=True)

    imgs = [im1, im2, im3]
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

elif filename == "compressionplot":
    # Plot compression ratio as a function of r for all images.
    filename1 = 'images/chessboard.png'
    filename2 = 'images/jellyfish.jpg'
    filename3 = 'images/new_york.jpg'
    im1 = imread(filename1, as_gray=True)
    im2 = imread(filename2, as_gray=True)
    im3 = imread(filename3, as_gray=True)

    imgs = [im1, im2, im3]
    
    # Save original images for compression calcs
    tmp_filenames = ['tmp/check_chessboard.png',
                     'tmp/check_jellyfish.png',
                     'tmp/check_new_york.png']
    plt.imshow(im1, cmap='gray')
    plt.savefig(tmp_filenames[0], format='png')
    plt.clf()
    plt.imshow(im2, cmap='gray')
    plt.savefig(tmp_filenames[1], format='png')
    plt.clf()
    plt.imshow(im3, cmap='gray')
    plt.savefig(tmp_filenames[2], format='png')
    plt.clf()
    
    ratios = [[] for im in imgs]
    r_vals = range(5, r, 5)
    for i in range(len(imgs)):
        for r_val in r_vals:
            comp, s = svd_compression_r(imgs[i], r_val)
            tmp_name = tmp_filenames[i] + "_compressed_"+str(r_val)+".png"
            plt.imshow(comp, cmap='gray')
            plt.savefig(tmp_name, format='png')
            plt.clf()

            ratios[i].append(get_compression_ratio(tmp_filenames[i],
                                                   tmp_name))


    # Plotting
    plt.plot(r_vals, ratios[0], 'ro', label=r'chessboard')
    plt.plot(r_vals, ratios[1], 'go', label=r'jellyfish')
    plt.plot(r_vals, ratios[2], 'bo', label=r'new york')
    plt.xlabel(r'$r$')
    plt.ylabel(r'Compression Ratio')
    plt.legend()
    plt.show()

elif filename == "nmcompression":
    # Plot compression ratio as a function of n and m for all pictures.
    # Should indicate how much image size factors in when compressing.
    filename1 = 'images/chessboard.png'
    filename2 = 'images/jellyfish.jpg'
    filename3 = 'images/new_york.jpg'
    im1 = imread(filename1, as_gray=True)
    im2 = imread(filename2, as_gray=True)
    im3 = imread(filename3, as_gray=True)

    imgs = [im1, im2, im3]
    
    # Save original images for compression calcs
    tmp_filenames = ['tmp/check_chessboard.png',
                     'tmp/check_jellyfish.png',
                     'tmp/check_new_york.png']
    plt.imshow(im1, cmap='gray')
    plt.savefig(tmp_filenames[0], format='png')
    plt.clf()
    plt.imshow(im2, cmap='gray')
    plt.savefig(tmp_filenames[1], format='png')
    plt.clf()
    plt.imshow(im3, cmap='gray')
    plt.savefig(tmp_filenames[2], format='png')
    plt.clf()
    
    ratios_n = [[] for im in imgs]
    ratios_m = [[] for im in imgs]
    
    # Stepsize 10 to avoid too long runtime.
    nm_vals = range(1, nm, 10)
    for i in range(len(imgs)):
        for n in nm_vals:
            comp, s = svd_compression_r(imgs[i][:-n,:], r)
            tmp_name = tmp_filenames[i] + "_compressed_"+str(n)+".png"
            plt.imshow(comp, cmap='gray')
            plt.savefig(tmp_name, format='png')
            plt.clf()

            ratios_n[i].append(get_compression_ratio(tmp_filenames[i],
                                                   tmp_name))

    for i in range(len(imgs)):
        for m in nm_vals:
            comp, s = svd_compression_r(imgs[i][:,:-m], r)
            tmp_name = tmp_filenames[i] + "_compressed_"+str(n)+".png"
            plt.imshow(comp, cmap='gray')
            plt.savefig(tmp_name, format='png')
            plt.clf()

            ratios_m[i].append(get_compression_ratio(tmp_filenames[i],
                                                     tmp_name))
    plt.plot(nm_vals, ratios_n[0], '-', label=r'chessboard n')
    plt.plot(nm_vals, ratios_n[1], '-', label=r'jellyfish n')
    plt.plot(nm_vals, ratios_n[2], '-', label=r'new york n')
    plt.plot(nm_vals, ratios_m[0], '-', label=r'chessboard m')
    plt.plot(nm_vals, ratios_m[1], '-', label=r'jellyfish m')
    plt.plot(nm_vals, ratios_m[2], '-', label=r'new york m')
    plt.title('Compression Ration vs. n and m')
    plt.xlabel(r'n and m')
    plt.ylabel(r'Compression Ratio')
    plt.legend()
    plt.show()


else:
    im = imread(filename, as_gray=True)
    comp, s = svd_compression_r(im, r)
    compname = filename[:-4] + "_compressed.png"
    non_compname = filename[:-4] + "_noncompressed.png"
    plt.imshow(im, cmap='gray')
    plt.savefig(non_compname, format='png')
    plt.clf()
    plt.imshow(comp, cmap='gray')
    plt.savefig(compname, format='png')
    size_comp = os.path.getsize(compname)
    size_noncomp = os.path.getsize(non_compname)
    print("Filesizes grayscale \nnon_compressed | compressed")
    print("{} MB | {} MB".format(size_noncomp*1e-6, size_comp*1e-6))
    print("Compression ratio = {}".format(get_compression_ratio(non_compname, compname)))
