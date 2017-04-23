#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* driveTraining.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* drive2_model.h5 and drive2_model.json containing a trained convolution neural network 
* drive2_model.h5 is uploaded online since it's large file... link https://drive.google.com/file/d/0B7fR3L0ypVR0aFRmX0Jwc1dLVzg/view?usp=sharing
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py drive2_model.json
```

####3. Submission code is usable and readable

The driveTraining.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network as defined in Nvidia model (driveTraining.ipynb) 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer . The model also includes cropping of images on the top and bottom of the images thereby reducing the unneccessary features for prediction. BatchNormalization helped reduce the internal covariate shift of each batch there by acting as a regularizer and reducing the need for dropouts. The batch normalization helped avoid model making extreme turns in the driving. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting by passing validation_split to Keras fit method. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.compile method).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVidia model. I thought this model might be appropriate because it has multiple convolution layers followed by fully connected layers. This architecture will learn the features of the road without any feature identification. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I added a callback to check for validation loss so that model stops training if the loss continues for 2 epochs. I added a history callback to check how the training and validation error progresses after each epoch, as seen in the picture, validation error initially decreases but increases again even though training error is decreasing, this is because the model is overfitting at this time. I had the algorithm to stop training when this happens.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I had to adjust steering correction for left and right images so that turning is neither too sharp or too straight. Flipping of images also helped train the model for both left and right turns.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (driveTraining.ipynb, create_model() function) consisted of cropping, image normalization, batch normalization, convolution neural network with the following layers and layer sizes ...

    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))    
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row, col,ch),output_shape=(row, col,ch)))    
    model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(nrows,ncols,3)))
    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

To augment the data sat, I also flipped images and angles thinking that this would help model to learn both left and right turns. Example flipped images are shown in notebook.

I also used images from left and right cameras with 0.15 steering correction. This also helped augment the dataset. 

After the collection process, I had 69672 number of data points. I then preprocessed this data by cropping and normalization.

I finally randomly shuffled the data set and put 0.1% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by graph in the notebook. I used an adam optimizer so that manually training the learning rate wasn't necessary.
