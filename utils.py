import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage import exposure

"""
Load an image as RGB
"""
def load_image(fname):
    return mpimg.imread(fname)


"""
Detect chessboard corners and return tuple of 3d real world space points with corresponding 2d image plane points
"""
def calc_chessboards_corners(images, num_x, num_y):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    objp = np.zeros((num_x * num_y, 3), np.float32)
    # [ 0.  0.  0.], [ 1.  0.  0.], [ 2.  0.  0.], ..., [ num_x-1.  num_y-1.  0.]
    # only for x, y corrdinates
    objp[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # img = cv2.drawChessboardCorners(img, (num_x, num_y), corners, ret)
            # plt.imshow(img)

    return objpoints, imgpoints


"""
Calibrate camera using passed world space points and corresponding image plane points.
The method returns Camera Matrix and Distortion Coefficients.
"""
def calibrate(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    return mtx, dist


"""
Undistort image using passed Camera Matrix and Distortion Coefficients.
"""
def undistort(img, calibration_mtx, calibration_dist):
    undist = cv2.undistort(img, calibration_mtx, calibration_dist, None, calibration_mtx)
    return undist


"""
Warp the image using passed Matrix
"""
def warp(img, matrix):
    img_size = (img.shape[1], img.shape[0])

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, matrix, img_size)

    return warped


"""
Draw lane on the image and warp it back using passed Inverse Matrix
"""
def unwarp(undist, Minv, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)
    return result


"""
Apply a Gaussian Noise kernel
"""
def gaussian_blur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

"""
Perform Histogram Equalization
"""
def equalize_hist(img):
    img = exposure.equalize_hist(img)
    img = (img * 255).astype(np.uint8)
    return img


