#**Behavioral Cloning**

---
This project was done as part of Udacity's Self-Driving Car Nanodegree Program. The model performance has been tested on for resolution of 640x480, and graphic quality selected as 'fastest'.

The model was only trained on track 1 data, to see the model performance click the following links:

* video.mp4

[//]: # (Image References)
[invidia_model]: ./images/invidia_model.jpeg "Nvidia Model"
[center_lane_driving]: ./images/center_lane_driving.jpg "Center Lane Driving"
[cropped_image]: ./images/cropped_image.jpeg "Cropped Image"
[center_left_right]: ./images/center_left_right.jpeg "Center Left Right"
[flipping_image]: ./images/flipping_image.jpeg "Flipping Image"


###Files Submitted & Code Quality

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

###Model Architecture and Training Strategy

####1. Solution Design Approach

Based on previous study course, There are a lot of neural network model, I want test some model, such as [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py),  [Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) ... . These model performance which I thought is very well, However my mac book pro with GPU (GeForce 540M) and CPU(2.7 GHz Intel Core i7) can't not handle some of these model. As a result, The final model architecture as same as nvidia (model.py lines 25-65) consisted of a convolution neural network. The Nvidia model as shown in the figure below:

![Nvidia Model][invidia_model]


####3. Creation of the Training Set & Training Process

#### Training and Validation Data Split

The data was split so as to use 80% of data as training set and 20% of the data was used for validation, and the validation data was used to ensure that the model does not overfit.

#### Data Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][center_lane_driving]

#### Cropping Image

The cameras in the simulator capture 160 pixel by 320 pixel images.
Not all of these pixels contain useful information, however. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. The code for the same is at [line 39 of utils.py](https://github.com/sumitbinnani/CarND-Behavioral-Cloning-P3/blob/master/utils.py#L18).

![Cropped Image][cropped_image]

#### Flipping Images And Steering Measurements

A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement. For example:

![Flipping Image][flipping_image]

#### Center, Left and Right Images

During training, I want to feed the left and right camera images to your model as if they were coming from the center camera. This way, I can teach your model how to steer if the car drifts off to the left or the right.

![Center Left Right][center_left_right]


#### Generator

Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.

