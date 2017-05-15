import matplotlib.image as mpimg
import numpy as np
import cv2


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


def undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def unwrap(img, objpoints, imgpoints, src, dst):
    undist_img = undistort(img, objpoints, imgpoints)

    img_size = (undist_img.shape[1], undist_img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist_img, M, img_size)

    return warped
