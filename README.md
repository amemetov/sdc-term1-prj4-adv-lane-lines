# **Advanced Lane Finding** 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration2]: ./output_images/calibration2.jpg "Chessboard"
[undistort_output]: ./output_images/undistort_output.png "Undistorted"
[undistort_test_img]: ./output_images/undistort_test_img.png "Undistorted Test Image"
[test_img2_thresholded]: ./output_images/test_img2_thresholded.png "Thresholded Test Image 2"
[test_img2_thresholded_gray]: ./output_images/test_img2_thresholded_gray.png "Gray Thresholded Test Image 2"
[warped_output]: ./output_images/warped_output.png "Warped"
[test_img2_warped]: ./output_images/test_img2_warped.png "Warped Test Image 2"
[test_img2_warped_thresholded]: ./output_images/test_img2_warped_thresholded.png "Warped and Thresholded Test Image 2"
[test_img2_fit]: ./output_images/test_img2_fit.png "2-nd Polynomial fit on Test Image 2"
[test_img2_fit_opt]: ./output_images/test_img2_fit_opt.png "2-nd Polynomial fit optimized method"
[test_img2_unwarp]: /output_images/test_img2_unwarp.png "Unwarp Test Image 2 with rendered found lane line"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Submitted Files

#### 1. Submission includes following files:

* [P4-Advanced-Lane-Finding.ipynb](P4-Advanced-Lane-Finding.ipynb) IPython notebook file containing workflow of the project
* [LaneLinesFinder.py](LaneLinesFinder.py) containing the entry point for lane line detecting
* [threshold.py](threshold.py) containing methods for images thresholding
* [utils.py](utils.py) containing helper methods
* [project_video_out.mp4](project_video_out.mp4) the result output video
* README.md summarizing the results


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell #2 of [the IPython notebook](P4-Advanced-Lane-Finding.ipynb).
Base job is done by method `calc_chessboards_corners` from [utils.py](utils.py) file.

First of all I prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
I assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
I use `cv2.findChessboardCorners` method to detect chessboards corners in a test image.
If all corners are detected then the `imgpoints` are appended with the (x, y) pixel position of each of the corners in the image plane.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


![alt text][undistort_output] |


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code cell #5 of [the IPython notebook](P4-Advanced-Lane-Finding.ipynb) contains a code which undistorts test image using `calibration_mtx` and `calibration_dist` variables computed in the code cell #2.


![alt text][undistort_test_img] |


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of 2 gradient thresholds (on Grayscaled image and S channel of the origin image) 
to generate a binary image.

I tried 2 approach to detect lane lines:
* Undistort -> Thresholding -> Warpping (method `threshold_origin_image` in [threshold.py](threshold.py))
* Undistort -> Warpping -> Thresholding (method `threshold_image` in [threshold.py](threshold.py))

For me the second approach gave a better result.
  
Here's an example of my output for the first approach:

![alt text][test_img2_thresholded_gray] |


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 60 in the file [utils.py](utils.py).  
The `warp()` function takes as inputs an image (`img`) and PerspectiveTransformMatrix `matrix`.
The `matrix` is computed in the IPython notebook in the cell #13 using hardcoded source (`src`) and destination (`dst`) points:

```python
src = np.float32([[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
                  [((img_size[0] / 6) + 10), img_size[1]],
                  [(img_size[0] * 5 / 6) + 60, img_size[1]],
                  [(img_size[0] / 2 + 75), img_size[1] / 2 + 100]])
dst = np.float32([[(img_size[0] / 4)-100, 0],
                  [(img_size[0] / 4)-100, img_size[1]],
                  [(img_size[0] * 3 / 4)+100, img_size[1]],
                  [(img_size[0] * 3 / 4)+100, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 220, 0        | 
| 223, 720      | 220, 720      |
| 1126, 720     | 1060, 720      |
| 715, 460      | 1060, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped_output]

For the curved examples I got this result:

![alt text][test_img2_warped]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find lane lines I used approach showed in the lecture *#33 Finding the Lines*.
I modified it, and used histogram of each strip (9 strips at all) to find the next windows which contains activated pixels.
The method `find_lane_lines` appears in line 331 in the file [LaneLinesFinder.py](LaneLinesFinder.py)

The I used a 2nd order polynomial to fit found pixels to the result curve.

Here is the result of Thresholding of Warped image:

![alt text][test_img2_warped_thresholded]


And below is the result of fitting 2nd order polynomial:


![alt text][test_img2_fit]


The method `optimized_find_lane_lines` which appears in the line 412 in the file [LaneLinesFinder.py](LaneLinesFinder.py)
is the method which tried to optimize lane lines finding process using already found lane lines.
Green boxes shows where the method looked for necessary pixels:

![alt text][test_img2_fit_opt]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The method `calc_curvature_and_dist` (line #223 in [LaneLinesFinder.py](LaneLinesFinder.py)) calculates the radius of curvature and the distance from the center of camera to the passed lane line.
I've used approach described in the lecture *#35. Measuring Curvature*.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the method `unwarp` line #72 in [utils.py](utils.py). 
Here is an example of my result on a test image:

![alt text][test_img2_unwarp]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To avoid jittering I implemented averaging lanel lines using recent 5 found lane lines - see method `average_fit` in line #192 in the file [LaneLinesFinder.py](LaneLinesFinder.py).

The most time I have spent to get good enough thresholded result.
Firstly I tried an approach showed in the lectures: Thresholding and then Warping.
But I did not manage to find values which perform well for all frames of project video.
Then I decided that Thresholding after Warping might give better result, I tried it and it worked after some parameters tuning.

But my solution does not work for [the challenge video](./challenge_video_out.mp4), and I think because thresholding finds a lot unnecessary gradients.
I think the same poor performance will be produced when some obstacles (cars, pedestrians, even shadows of trees, bridges, buildings) will appear on the road.

Performance is not appropriate too, cause it's not realtime. Profiling is required and further fixing/modifying.
