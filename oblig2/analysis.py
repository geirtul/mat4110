import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# Load images
"""
Images fetched from links provided in task:
% https://pixabay.com/en/board-chess-chessboard-black-white-157165/
% https://pixabay.com/en/jellyfish-under-water-sea-ocean-698521/
% https://pixabay.com/en/new-york-city-skyline-nyc-690868/
"""

im1 = imread('images/chessboard.png', as_gray=True)
im2 = imread('images/jellyfish.jpg', as_gray=True)
im3 = imread('images/new_york.jpg', as_gray=True)

print(np.amax(im1))

