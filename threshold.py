import numpy as np
import cv2


# This function applies Sobel x or y, then takes an absolute value and applies a threshold.
def abs_sobel_threshold(sobel, thresh=(0, 255)):
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return this mask as binary_output image
    return binary_output


# This function applies Sobel x and y, then computes the magnitude of the gradient and applies a threshold
def mag_threshold(sobelx, sobely, thresh=(0, 255)):
    # Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx * sobelx + sobely * sobely)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return this mask as binary_output image
    return binary_output


# This function applies Sobel x and y, then computes the direction of the gradient and applies a threshold.
def dir_threshold(sobelx, sobely, thresh=(0, np.pi / 2)):
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction > thresh[0]) & (grad_direction < thresh[1])] = 1
    # Return this mask as binary_output image
    return binary_output


def gradient_threshold(img, working_ch='gray',
                       ksize=3, x_abs_thresh=(0, 255), y_abs_thresh=(0, 255),
                       mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2)):
    if working_ch == 'L':
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        one_ch_img = hsv[:, :, 1]
    elif working_ch == 'S':
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        one_ch_img = hsv[:, :, 2]
    else:
        # by default grayscale is used
        one_ch_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(one_ch_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(one_ch_img, cv2.CV_64F, 0, 1, ksize=ksize)

    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(sobelx, thresh=x_abs_thresh)
    # grady = abs_sobel_threshold(sobely, thresh=y_abs_thresh)
    mag_binary = mag_threshold(sobelx, sobely, thresh=mag_thresh)
    dir_binary = dir_threshold(sobelx, sobely, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def threshold_image(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    ksize = 9
    x_abs_thresh = (50, 100)
    y_abs_thresh = (50, 100)
    mag_thresh = (50, 100)
    dir_thresh = (0.7, 1.3)
    sxbinary = gradient_threshold(img, 'S', ksize, x_abs_thresh, y_abs_thresh, mag_thresh, dir_thresh)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    # l_channel = hsv[:,:,1]
    s_channel = hsv[:, :, 2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine
    combined = np.zeros_like(sxbinary)
    combined[(sxbinary == 1) | (s_binary == 1)] = 1

    # Stack each channel
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.dstack((combined, sxbinary, s_binary))
    return color_binary