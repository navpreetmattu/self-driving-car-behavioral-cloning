# **Behavioral Cloning** 

## Writeup Template


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA_Architecture.png "Model Visualization"
[image2]: ./examples/center.jpg "Grayscaling"
[image3]: ./examples/right.jpg "Recovery Image"
[image4]: ./examples/left.jpg "Recovery Image"
[image5]: ./examples/normal.jpg "Normal Image"
[image6]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the *drive.py* file, the car can be driven autonomously around the track by executing below line. Here, *model.h5* is the trained model which predicts the steering angle.
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The *model.py* file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used the neural network model architecture provided by NVIDIA. The model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64. It is defined in method **nvidia_model()** (model.py lines 67-87) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with keeping probability between 60-80% in order to reduce overfitting (code lines 78, 80, 82, 84). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also, I drove the vehicle in opposite direction to reduce the left side bias and get more data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the *Nvidia Self Driving Car Model Architecture*. I thought this model might be appropriate because it is providing very promising results in the paper published by Nvidia.

In order to gauge how well the model was working, I split image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout in dense layers of the neural network with a keep probability of 60-80%.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like when the turn is very sharp. To improve the driving behavior in these cases, I collected more data and recorded the vehicle recovering from the left side and right sides of the sharp turns back to center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-87) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	                         | 
|:---------------------:|:------------------------------------------:| 
| Input         		| 160x320x3 RGB image   		             | 
| Cropping         		| 160x320x3 -> 70x320x3   		             | 
| Lambda         		| Normalize Images b/w -1 & 1   		     | 
| Convolution 5x5     	| 2x2 stride, valid padding, ELU Activation |
| Convolution 5x5     	| 2x2 stride, valid padding, ELU Activation |
| Convolution 5x5     	| 2x2 stride, valid padding, ELU Activation |
| Convolution 3x3     	| 1x1 stride, valid padding, ELU Activation |
| Convolution 3x3     	| 1x1 stride, valid padding, ELU Activation |
| Fully connected		| 1164 nodes        						 |
| Dropout               | 60 % keep probabolity                      |
| Fully connected		| 100 nodes        						     |
| Dropout               | 70 % keep probabolity                      |
| Fully connected		| 50 nodes        						     |
| Dropout               | 80 % keep probabolity                      |
| Fully connected		| 10 nodes        						     |
| Dropout               | 60 % keep probabolity                      |
| Fully connected		| 1 nodes        						     |

Here is a visualization of the architecture (Source: NVIDIA)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center of the road from the sides. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]

Then I took a U-turn and recorded a lap on the opposite ide of the road. To make the training data more generalized.

To augment the data sat, I also flipped images and angles thinking that this would help in reducing the left steer bias. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had X number of data points, which also contains the data points from the test samples provided by udacity. I, then, cropped the top and bottom portion of the images, since they are not giving us any valuable information which can help train to drive the car. Then, I normalized the image by dividing each pixel value by 255 and subtracting .5.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I have incremently trained the model. First with udacity data for 5 epochs and second time with the data collected by me for 5 epochs. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. 

After all the training, I put the car on autonomous mode and it was driving very good by itself.
