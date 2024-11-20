# Case study/ Project: Detection of Melanoma - by Saunak MALLIK
> Description: This assignment is a Deep Learning model building assignment to build a Convolutional Neural Network (CNN) model that would aid dermatologists to detect the presence of melanoma from the image of the lesion.

> Note: This case study is for learning/ educational purposes only.

>  #Heathcare #Diagnostic #Melanoma
## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
- __Domain__. Healthcare Industry.

- __Background__. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

- __Dataset__. ISIC dataset is used in this project. The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


- __Dataset classes__.The data set contains the following diseases:
    - Actinic keratosis
    - Basal cell carcinoma
    - Dermatofibroma
    - Melanoma
    - Nevus
    - Pigmented benign keratosis
    - Seborrheic keratosis
    - Squamous cell carcinoma
    - Vascular lesion

- __Final outcome of the model__. We have built a multiclass classification model using a custom Convolutional Neural Network (CNN) in TensorFlow, which can accurately detect the presence of melanoma or Skin cancer based on the features of lesions from images. Early detection would lead to early treatment of melonema.

    Note: Transfer Learning has not been used.

## Project pipeline and Conclusions
- __Data Reading/Data Understanding__: Defined the path for train and test images 
- __Dataset Creation__. Create train & validation dataset from the train directory with a batch size of 32. Also, resized the images to 180*180 pixels.
- __Dataset visualisation__: Created code to visualize one instance each of all the nine classes present in the dataset.
- __Model Building & training__: Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescaled the images to normalize pixel values between (0,1).
- __Optimiser and LOSS Function__. I have choose __ADAM__ optimiser and __Sparse Categorical Crossentropy__ loss function for model training. I have trained the model for __~20 epochs__.
- At each step, I have build a CNN model, compiled it, summarised it and then trained it. I have used __RELU__ activation function and __SOFTMAX (because of 9 classes - Multi-class classification problem)__ in the last layer.
- __MODEL BUILDING__. I have created 3 models -

    - __Model1__ . Base model with training and testing accuracy of 92% and 55% shows the presence of very high __OVERFITTING__ in Model1  i.e., the model may not generalize well to unseen data.
    - Train the model for 20 epochs.
    - Conclusion. Training accuracy is 92.9% while the Validation accuracy is 55.7% indicating presence of OVERFITTING in Model1 i.e., the model may not generalize well to unseen data.
    - The graph reflects that -
        - During the training process, the model's training accuracy linearly increases over time, whereas the validation accuracy plateaus at 50% accuracy.
        - As the training epochs progress, the training loss decreases, however, the validation loss exhibits an opposing trend and increases.


    - __Model2__ . Enrich the previous model by __Augmenting the dataset__ (training, validation and testing) with __RandomFlip, RandomRotation and RandomZoom__ and adding a __DROPOUTS layer__ additionally.
    - Train the model for 20 epochs.
    
    - Conclusion. Training accuracy is 64.9% while the Validation accuracy is 53% still indicating presence of slight OVERFITTING as compared with Model1; addition of DROPOUTS hasn't addressed OVERFITTING much i.e., the model may not generalize well to unseen data.
    - The graph reflects that -
        - During the training process, the model's training accuracy linearly increases over time and gets stalled around 55%-65%, whereas the validation accuracy plateaus at 50%-53% accuracy.
        - As the training epochs progress, the training loss decreases, however, the validation loss stalls around 1.5.
    
    - __Model3__. __Rectify the Class Imbalance and re-run Model2. 
    - Train the model for 30 epochs.
    - Conclusion. With Training accuracy is 84.6% and Validation accuracy of 78.8%, the OVERFITTING problem has been RESOLVED in Model3 i.e., the Model should perform well on Unknown dataset.
    - Thus, we can conclude that Data Augmentation technique and Class Imbalance handling has played a significant Role in addressing the the overfitting problem initially present in the Dataset while also Improving the Model accuracy.
    - The graph reflects that as the training epochs progresses during the Training process -
        - the model's Training and Validation accuracy linearly increases over time.
        - the Training loss and Validation loss decreases linearly.

- __Business Conclusions__. The multiclass classification model using a custom Convolutional Neural Network (CNN) in TensorFlow, can accurately detect the presence of melanoma or Skin cancer based on the features of lesions from images. Early detection would lead to early treatment of melonema.

## Technologies Used
- python 3 with Jupyter Note Book
- libraries used - sklearn, statsmodels.api, numpy, pandas, warnings
- libraries used for data visualization - matplotlib, seaborn
- Tensorflow and Keras libraries for CNN Model building
- Keras and Augmentor libraries for Data Augmentation


## Acknowledgements

- This project and the data was provided by [IIITB] (https://www.iiitb.ac.in/) & [upGrad](https://www.upgrad.com/) learning platform.
- Please note that the dataset is for learning purposes only.


## Contact
- Created by [Saunak MALLIK]
(https://github.com/saunakmallik2502/)- feel free to contact me!
- Project work repository : [Melanoma Detection using CNN](https://github.com/saunakmallik2502/Deep_Learning)).
