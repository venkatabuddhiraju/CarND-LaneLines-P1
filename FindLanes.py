#!/usr/bin/env python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import HelperFunctions as util
import numpy as np
import cv2


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

imshape = image.shape
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
#plt.show()



util.line_segments(image)
#util.process_all_images()