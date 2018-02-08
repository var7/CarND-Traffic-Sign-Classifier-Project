**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[distbefore]: ./visualization/dist_before_aug.png
[100explore]: ./visualization/explore.png
[move]: ./visualization/move.png
[rot]: ./visualization/rot.png
[scale]: ./visualization/scale.png
[distafter]: ./visualization/distafter.png
[gray]: ./visualization/gray.png
[norm]: ./visualization/norm.png
[web]: ./visualization/web.png
[webpred]: ./visualization/webpred.png
[validcurve]: ./visualization/validcurve.png

---
### README

#### Link to the [project code](https://github.com/var7/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Load, Explore, summarize the dataset

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![original distribution][distbefore]

As we can see the data is not distributed equally in all the classes. This can cause problems during prediction because the classifier would have seen fewer samples of a particular class. 


### Design and Test a Model Architecture

#### Dataset Augmentation
As seen above, in the dataset the number of examples in each class is not equally distributed. To fix this and add to the number of training examples I decided to augment the dataset. It also helped improve the performance of my network - taking it from somewhere around 89% to 95%.

If there were any classes that had less than 800 samples, I augmented the data in that class till there were 800 samples. This was done by first moving the image, then rotating it, and then scaling it. I moved the images by a maximum of 4px. For rotation the range was 15 degrees. For scaling I allowed the image to be scaled by upto 6px (about 18% zooming in or out).

Rotated by: -4.38 deg
![rotation][rot]
Moved to x:0.46 y:-2.46 from (0,0)
![move][move]
Scaled by : 3.64 px
![scale][scale]

After the augmentation the distribution of the training set was more equally distributed. This is shown in the below bar graph

![after augmentation][distafter]

#### Preprocessing of image

As a first step, I decided to convert the images to grayscale because LeNet obtained good results after grayscaling. Even the Sermanet paper performed grayscaling. In my tests, I found that it did not make much of a difference if we grayscaled or not. It also resulted in slightly faster performance. So I decided to stick with it

Here is an example of a few traffic sign images after grayscaling.

![gray][gray]

As a last step, I normalized the image data mostly because the lessons told me to do so. However, after normalization some of the images which were too dark or too bright looked better.

Here is an example of applying normalization to a few grayscaled images.

![norm][norm]


#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| ELU					| ELU activation								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| ELU					| ELU activation								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Flatten the conv output, outputs 400			|
| Fully connected		| Fully connected, outputs 120					|
| ELU					| ELU activation								|
| Fully connected		| Fully connected, outputs 84					|
| ELU					| ELU activation								|
| Fully connected		| Fully connected, outputs 43					|
| softmax  				| Outputs probabilities for 43 classes			|
 
I decided to use a different activation function from ReLU. I used the [ELU activation function](https://arxiv.org/abs/1511.07289). The authors show that ELU performs better across different datasets.

#### Training the model

To train the model, I used a batch size of 64 and 150 epochs. I found that a learning rate of 0.001 gave good results. I used an ADAM optimizer. 

The number of epochs (x-axis) vs validation error (y-axis) plot is given below
![validation curve][validcurve]

#### Final results

My final model results were:
* validation set accuracy of 95.6% (maximum)
* test set accuracy of 93.4%
 

###Test a Model on New Images

#### Images chosen
Here are the five German traffic signs that I found on the web:

![web][web]

The first and fourth images might be difficult to classify because they are slightly rotated along the vertical axis. 

#### Web images test results

Here are the results of the prediction:

| Image			        					|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| 30 km/h      								| 30 km/h   									| 
| 50 km/h    								| 50 km/h 										|
| 70 km/h									| 70 km/h 										|
| Turn left ahead	    					| Turn left ahead					 			|
| Right-of-way at the next intersection		| Right-of-way at the next intersection			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.4% 
However, this is not necessarily true that the model performs better than expected. Many of the images in the dataset provided had poor contrast, low brightness or were blown out. The images that I could find on the internet were mostly well exposed, and had good contrast.

#### Looking at top 5 predictions for each of the web images

The images and the corresponding top 5 prediction classes are shown below. As an illustration of the class predicted an image from that class in the dataset is shown. 
![webprediction][webpred]
Some of the classes have a greater than 100% probability prediction. I am not too sure what that means. Most of the other predictions in the top 5 share similarities like shape of sign,  and sometimes numbers. 

