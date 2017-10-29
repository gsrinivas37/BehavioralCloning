#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[network]: ./examples/network.png "Network architecture"
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have used an architecture published by NVidia self driving car team linked below.
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

The model consists of a five convolution layers followed by three fully connected layers. First 3 convolutional layers have 5x5 kernel with 2x2 stride and next two layers have 3x3 kernel size without any stride.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

To avoid overfitting, 
1. I augmented the data with flipping the images so that the model learns to steer to the left and the right.
2. I used images from all three angles (left, centre and right) and used correction factor of 0.2.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The followed the below steps to finally arrived at my final solutions.

1. I started with single layer architecture to train the model. I used the sample training data and created the model.h5 file and used to test that it drives the simulator in Autonomous mode. It did work even though it didn't drive well but it provided me starting point to verify that I can train a model and use it to drive the vehicle.
2. Next I tried to use the well-knows LeNet architecture and have done some pre-processing of input data.
3. I have divided the pixels by 255 and subtracted 0.5 so that the values are centered around 0 with a deviation of 0.5
4. I augemented the input samples by flipping each of the image and negating the steering angle. This way the model learns to steer in both directions.
5. I have removed the upper 70 and lower 25 pixels which contains trees, mountains, car hood, etc which distracts the model to learn properly.
6. With above changes, the model was able to drive the car much better. It was not perfect but it was very close.
7. Next I tried to use all the three images from different cameras. Earlier I was using only center camera image. I used correction factor of 0.2 to force the car to steer to the right/left based on whether the image is from left or right camera.
8. Also I used a better network architecture published by NVidia self driving car team.
9. With the above changes, the model was able to drive the car on the road without going beyond the borders.
10. Later I generated more data by driving the simulator by driving around corners, sterring to the centre from the sides to finetune the model even better.

####2. Final Model Architecture

Here is a visualization of the architecture 

![alt text][network]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I followed below strategy.

1. I recorded the training for close to two loops driving mostly in the centre of the lane.
2. I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the centre if it even crossed the border.
3. I also recorded more data around the corners at slower speed because I wanted the model to have more data around the corners.
4. I used the recorded data in conjunction with data from https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

To augment the data sat, I also flipped images so that I can easily double the number of images and also the model learns to steer both to left and right.

After the collection process, I had 13107*6 number of data points. I then preprocessed this data by
1. Normalizing the values by dividing by 255 and substracted by 0.5
2. Removed the top 70 and bottom 25 pixels to remove unnecessary distractions.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
