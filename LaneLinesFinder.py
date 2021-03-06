import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from collections import deque

import utils
import threshold


"""
Class holds the characteristics of each line detection
"""
class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        self.recent_fits = deque([], maxlen=5)

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None


"""
LaneLinesFinder class manages lane detecting process.
It requires Camera calibration data (3d real world space points with corresponding 2d image plane points)
and warp/unwarp points.
The entry point of the class is the method process_image.
"""
class LaneLinesFinder(object):
    def __init__(self, distortion_obj_pts, distortion_img_pts, warp_src_pts, warp_dst_pts):
        self.distortion_obj_pts = distortion_obj_pts
        self.distortion_img_pts = distortion_img_pts
        self.warp_mtx = cv2.getPerspectiveTransform(warp_src_pts, warp_dst_pts)
        self.warp_mtx_inv = cv2.getPerspectiveTransform(warp_dst_pts, warp_src_pts)
        self.calibration_mtx = None
        self.calibration_dist = None

        self.left = Line()
        self.right = Line()

        self.recent_dist_min_y = 0
        self.recent_dist_mid_y = 0
        self.recent_dist_max_y = 0

    """
    Method process_image do the following steps:
    1. Computes Camera Calibration Matrix and Distortion Coefficients (only for the first passed image)
    2. Undistorts the image
    3. Warps the image to get top-down-perspective (aka bird-eye-view)
    4. Thresholds the image
    5. Detects lane line (using optimized version if possible)
    6. Computes average lane line using recent detected lanes.
    6. Computes a curvature and a position relatively to the lane.
    7. Draws detected lane line on the undistorted image
    8. Draws the curvature an the position info on the undistorted image
    9. Returns the result undistorted image
    """
    def process_image(self, img):
        if self.calibration_mtx is None:
            self.calibration_mtx, self.calibration_dist = utils.calibrate(img, self.distortion_obj_pts, self.distortion_img_pts)
            #mpimg.imsave('1.jpg', img, format='jpg')

        undist_img = utils.undistort(img, self.calibration_mtx, self.calibration_dist)

        ## origin_image -> undistort -> threshold -> top-down-perspective (aka bird-eye-view)
        #thresholded_img = threshold.threshold_origin_image(undist_img)
        #warped_img = utils.warp(thresholded_img, self.warp_mtx)[:,:,0]
        #out_img, left_fit, right_fit = self.find_lane_lines(warped_img)

        # origin_image -> undistort -> top-down-perspective (aka bird-eye-view) -> threshold
        warped_img = utils.warp(undist_img, self.warp_mtx)
        thresholded_img = threshold.threshold_image(warped_img)
        out_img, left_fit, right_fit = self.find_lane_lines(undist_img, thresholded_img[:, :, 0])

        return self.post_process_frame(undist_img, left_fit, right_fit)

    """
    Detects lane lines.
    Tries to use optimized method if possible.
    If lane line detected by optimized version has unexpected structure
    (distances between left and right lines have large difference along the lines),
    then method uses straight method.
    """
    def find_lane_lines(self, undist_img, warped_img):
        if self.left.detected == False or self.right.detected == False:
            return find_lane_lines(warped_img, gen_out_img=False)

        out_img, left_fit, right_fit = optimized_find_lane_lines(warped_img, self.left.current_fit, self.right.current_fit, gen_out_img=False)

        # check whether optimized version was managed to find lane lines
        if left_fit is None or right_fit is None:
            # try straight method
            return find_lane_lines(warped_img, gen_out_img=False)

        # Sanity checks
        dist_min_y, dist_mid_y, dist_max_y = self.calc_distances(warped_img, left_fit, right_fit)

        max_dist_diff = 50
        if math.fabs(dist_min_y - dist_mid_y) > max_dist_diff or \
            math.fabs(dist_min_y - dist_max_y) > max_dist_diff or \
            math.fabs(dist_mid_y - dist_max_y) > max_dist_diff:
            # try straight method
            return find_lane_lines(warped_img, gen_out_img=False)

        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undist_img, 'Dist min Y {0:.2f}'.format(dist_min_y), (600, 50), font, 1, txt_color, 2)
        cv2.putText(undist_img, 'Dist mid Y {0:.2f}'.format(dist_mid_y), (600, 100), font, 1, txt_color, 2)
        cv2.putText(undist_img, 'Dist max Y {0:.2f}'.format(dist_max_y), (600, 150), font, 1, txt_color, 2)

        return out_img, left_fit, right_fit

    """
    Calculates distances between left and right lines (at the beginning, at the middle, at the end)
    """
    def calc_distances(self, img, left_fit, right_fit):
        ploty, left_fitx, right_fitx = generate_fit_coords(img, left_fit, right_fit)
        min_y, max_y = np.min(ploty), np.max(ploty)
        mid_y = (min_y + max_y) / 2

        poly_left = np.poly1d(left_fit)
        poly_right = np.poly1d(right_fit)
        dist_min_y = math.fabs(poly_left(min_y) - poly_right(min_y))
        dist_mid_y = math.fabs(poly_left(mid_y) - poly_right(mid_y))
        dist_max_y = math.fabs(poly_left(max_y) - poly_right(max_y))
        return dist_min_y, dist_mid_y, dist_max_y

    """
    Stores info about found lane.
    Builds average lane using recent detected lanes.
    Calculates the curvature and the position.
    Draws info and detected lane on the undistorted image.
    """
    def post_process_frame(self, undist_img, left_fit, right_fit):
        # store lines
        self.left.detected = left_fit is not None
        self.right.detected = right_fit is not None

        self.left.current_fit = left_fit
        self.right.current_fit = right_fit

        # build average fit
        avg_left_fit = self.average_fit(self.left, left_fit)
        avg_right_fit = self.average_fit(self.right, right_fit)

        # store fits
        if left_fit is not None and right_fit is not None:
            self.left.recent_fits.append(left_fit)
            self.right.recent_fits.append(right_fit)

        # use average fits
        left_fit = avg_left_fit
        right_fit = avg_right_fit

        ploty, left_fitx, right_fitx = generate_fit_coords(undist_img, left_fit, right_fit)

        # calc curvature
        self.left.radius_of_curvature, self.left.line_base_pos = calc_curvature_and_dist(undist_img, ploty, left_fitx)
        self.right.radius_of_curvature, self.right.line_base_pos = calc_curvature_and_dist(undist_img, ploty, right_fitx)
        deviation_from_center = (math.fabs(self.right.line_base_pos - self.left.line_base_pos)/2)*100

        self.print_info(undist_img, left_fit, right_fit, deviation_from_center)

        result = utils.unwarp(undist_img, self.warp_mtx_inv, ploty, left_fitx, right_fitx)
        return result

    """
    Draws info on the undistorted image
    """
    def print_info(self, undist_img, left_fit, right_fit, deviation_from_center):
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undist_img, 'Left radius curvature: {0:.2f} m'.format(self.left.radius_of_curvature), (10, 50), font, 1, txt_color, 2)
        cv2.putText(undist_img, 'Right radius curvature: {0:.2f} m'.format(self.right.radius_of_curvature), (10, 100), font, 1, txt_color, 2)
        cv2.putText(undist_img, 'Distance to Left line: {0:.2f} m'.format(self.left.line_base_pos), (10, 150), font, 1, txt_color, 2)
        cv2.putText(undist_img, 'Distance to Right line: {0:.2f} m'.format(self.right.line_base_pos), (10, 200), font, 1, txt_color, 2)
        cv2.putText(undist_img, 'Deviation from center: {0:.2f} cm'.format(deviation_from_center), (10, 250), font, 1, txt_color, 2)
        #cv2.putText(undist_img, 'Left: {0}'.format(left_fit), (10, 300), font, 1, txt_color, 2)
        #cv2.putText(undist_img, 'Right: {0}'.format(right_fit), (10, 350), font, 1, txt_color, 2)

    """
    Builds average fit using recent detected fits
    """
    def average_fit(self, line, fit):
        models_num = len(line.recent_fits)
        if models_num == 0:
            return fit

        avg_models_weights_start = 0.01
        avg_models_weights_stop = 0.99
        weights = np.linspace(avg_models_weights_start, avg_models_weights_stop, num=models_num)
        idx = 0
        coeffs_accum = None
        for recent_fit in line.recent_fits:
            coeffs = np.copy(recent_fit) if recent_fit is not None else np.zeros(3)
            coeffs *= weights[idx]
            if coeffs_accum is None:
                coeffs_accum = coeffs
            else:
                coeffs_accum += coeffs
            idx += 1

        if fit is not None:
            coeffs_accum += fit
            result_coeffs = coeffs_accum / (np.sum(weights) + 1)
        else:
            result_coeffs = coeffs_accum / np.sum(weights)

        return result_coeffs


"""
Calculate the curvature of the line and distance from the (top, center) of the image to the line
"""
def calc_curvature_and_dist(img, ploty, fitx):
    # Define y-value where we want radius of curvature
    # choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    radius_of_curvature = ((1 + (2 * fit[0] * y_eval * ym_per_pix + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    poly = np.poly1d(fit)

    line_x = poly(y_eval * ym_per_pix)
    camera_x = (img.shape[1] / 2.) * xm_per_pix

    distance = math.fabs(line_x - camera_x)

    return radius_of_curvature, distance

"""
Generates (x,y) points for left and right fits
"""
def generate_fit_coords(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return ploty, left_fitx, right_fitx

"""
Detects initial peaks
"""
def find_initial_left_right_positions(img):
    w, h = img.shape[1], img.shape[0]

    # Take a histogram of the bottom half of the image
    #histogram = np.sum(img[h // 2:, :], axis=0)
    histogram = np.sum(img[h // 2 - 100:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    mid_x = w // 2
    left_x = np.argmax(histogram[:mid_x])
    right_x = np.argmax(histogram[mid_x:]) + mid_x
    return left_x, right_x

"""
Detects peaks at the passed range [y_low, y_high].
The peak value is the largest value found in the pointed left and right windows.
"""
def find_peaks(img, left_x_curr, right_x_curr, y_low, y_high,
               left_win_x_low, left_win_x_high, right_win_x_low, right_win_x_high, min_pix=1):
    w, h = img.shape[1], img.shape[0]
    histogram = np.sum(img[y_low:y_high, :], axis=0)

    mid_x = w // 2
    left_histogram = histogram[:mid_x]
    right_histogram = histogram[mid_x:]
    hist_left_indices = np.argsort(left_histogram)
    hist_right_indices = np.argsort(right_histogram)

    left_peak = left_x_curr
    right_peak = right_x_curr

    for ind in reversed(hist_left_indices):
        if histogram[ind] < min_pix:
            # keep curr pos
            break
        if ind >= left_win_x_low and ind <= left_win_x_high:
            # found max peak in win
            left_peak = ind
            break

    for ind in reversed(hist_right_indices):
        # adjust ind
        ind = ind + mid_x
        if histogram[ind] < min_pix:
            # keep curr pos
            break
        if ind >= right_win_x_low and ind <= right_win_x_high:
            right_peak = ind
            break

    return left_peak, right_peak

"""
Computes window left and right bounds
"""
def calc_win_bounds(left_x_curr, right_x_curr, margin):
    left_win_x_low = left_x_curr - margin
    left_win_x_high = left_x_curr + margin
    right_win_x_low = right_x_curr - margin
    right_win_x_high = right_x_curr + margin
    return left_win_x_low, left_win_x_high, right_win_x_low, right_win_x_high


"""
Detects lane lines.
img is the binary warped image (Top-Down Perspective Mapping).
The method splits the image on n_wins strides.
For the first stride find_initial_left_right_positions method is used to find starting windows positions.
For the next strides find_peaks method is used to detect peak on the stride.
The second polynomial is used to fit the found pixels.
"""
def find_lane_lines(img, gen_out_img=True, n_wins=9, win_w=150):
    #print('find_lane_lines image.shape: {0}'.format(img.shape))
    w, h = img.shape[1], img.shape[0]

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))*255

    # Set height of windows
    win_h = h // n_wins

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    margin = win_w // 2

    left_x_curr, right_x_curr = 0, 0
    left_win_x_low, left_win_x_high, right_win_x_low, right_win_x_high = 0, 0, 0, 0
    for win in range(0, n_wins):
        win_y_high = h - win * win_h
        win_y_low = win_y_high - win_h
        #print('win: ({0}, {1})'.format(win_y_low, win_y_high))

        if win == 0:
            left_x_curr, right_x_curr = find_initial_left_right_positions(img)
            left_win_x_low, left_win_x_high, right_win_x_low, right_win_x_high = calc_win_bounds(left_x_curr, right_x_curr, margin)
        else:
            left_x_curr, right_x_curr = find_peaks(img, left_x_curr, right_x_curr, win_y_low, win_y_high, left_win_x_low, left_win_x_high, right_win_x_low, right_win_x_high)
            left_win_x_low, left_win_x_high, right_win_x_low, right_win_x_high = calc_win_bounds(left_x_curr, right_x_curr, margin)

        if gen_out_img == True:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (left_win_x_low, win_y_low), (left_win_x_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (right_win_x_low, win_y_low), (right_win_x_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= left_win_x_low) & (nonzerox < left_win_x_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= right_win_x_low) & (nonzerox < right_win_x_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None

    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None

    if gen_out_img == True:
        # mark left and right lane indices
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fit, right_fit


"""
The optimized lane lines find method.
The method uses previous lane line to detect a lane line on the warped_image.
"""
def optimized_find_lane_lines(warped_img, left_fit, right_fit, gen_out_img=True):
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
    (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    if gen_out_img == True:
        # Generate x and y values for plotting
        ploty, left_fitx, right_fitx = generate_fit_coords(warped_img, left_fit, right_fit)

        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return out_img, left_fit, right_fit


"""
Lane line find method from lecture
"""
def find_lane_lines2(binary_warped, gen_out_img=True):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None

    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None

    # mark left and right lane indices
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fit, right_fit#, left_lane_inds, right_lane_inds, nonzerox, nonzeroy





def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_lane_lines2(warped):
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

        # Extract left and right line pixel positions
        print(np.max(l_points))
        leftx = l_points[0]
        lefty = l_points[1]
        rightx = r_points[0]
        righty = r_points[1]

        # Fit a second order polynomial to each
        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = None

        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = None

        #return output, left_fit, right_fit
        return output, None, None

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)
        return output, None, None


def find_window_centroids(warped, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids



