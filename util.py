from skimage import color
import numpy as np
import cv2

def preprocess_image_gray(image):
	# crop image in y direction: 67 to 132
	# resize image in x and y direction: half resolution
	image = image[67:132:4, 0:-1:4, :]
	shape = image.shape
	# convert rgb to gray
	image = color.rgb2gray(image).reshape(shape[0], shape[1], 1);
	# normalize value
	image = image / 255.0 * 0.8 + 0.1
	return image

def preprocess_image_r(image):
	# crop image in y direction: 67 to 132
	# resize image in x and y direction: half resolution
	image = image[65:135:4, 0:-1:4, 0]
	shape = image.shape
	image = image.reshape(shape[0], shape[1], 1);
	# normalize value
	image = image / 255.0 - 0.5
	return image

def preprocess_image_sobel(image):
	kernel = 3
	min_th = 15
	max_th = 100
	image = image[65:135, :, :]
	# Convert image
	converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(converted[:,:,1], cv2.CV_64F, 1, 0, ksize=kernel)
	sobely = cv2.Sobel(converted[:,:,1], cv2.CV_64F, 0, 1, ksize=kernel)
	# Rescale images
	sobelx = sobelx[0:-1:4, 0:-1:4]
	sobely = sobely[0:-1:4, 0:-1:4]
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= min_th) & (gradmag <= max_th)] = 1
	shape = binary_output.shape
	binary_output = binary_output.reshape(shape[0], shape[1], 1)
	return binary_output

def preprocess_image_canny(image):
	kernel = 3
	min_th = 0
	max_th = 255
	image = image[65:135:4, 0:-1:4, :]
	# Convert image
	converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	# Blur image
	blurred = cv2.GaussianBlur(converted[:,:,1], (kernel, kernel), 0)
	# Calculate the gradient magnitude
	edges = cv2.Canny(blurred, min_th, max_th)
	# Rescale to 8 bit
	scale_factor = np.max(edges)
	edges = (edges / scale_factor).astype(np.uint8)
	shape = edges.shape
	edges = edges.reshape(shape[0], shape[1], 1)
	return edges

def preprocess_image(image):
	return preprocess_image_sobel(image)

def flip_image(image):
	shape = image.shape
	flipped = cv2.flip(image, 1)
	flipped = flipped.reshape(shape)
	return flipped