# AI_ML_Capston_Project

## Capston Project for SCTP AI-ML Associate Developer Course - Image Classification using Deep Neural Network

# Introduction

Artificial Intelligence is achieved by applying Machine learning and Deep learning models to classification, identification, prediction and generation of text, vision, speech.
 Human vision processing is highly evovled. Comparatively, AI based computer vision is in progressing stages. Basic building blocks of computer vision system involves classification (object recognition), object detection, instance segementation, object location and so on.
 <>

**Problem statement:** Correct image classification (or object recognition) is vital as it leads to next steps in computer vision.
This could either be achieved by tabulating distinguishing attributes of objects to classify, like length and width attributes of Iris flower species in Iris dataset. Or, by extracting and identifying the edges and features from the binary pixel data of the digital images.

 Former uses the unsupervised (classification) machine learning models, like Logisitcs Regression, KNN, SVM, Decision Tree Classifier and Random Forest.

 Latter is performed using deep learning neural networks, specifically **Convolutional Neural Networks (CNNs)** and/or variants.

**Project Purpose:**
Learn and develop a CNN based image classification system using Tensorflow and Keras frameworks. Additionally, familarize with the existing image classification research and models, and the next steps in the computer vision.

# Literature Review

### a. Image Classification Models:

  As a legend, computer vision started with the **MIT Professor Marvin Minsky**'s 1960 summar project assignment to undergraduates, that required them to attach a camera to a computer and have the computer describe everything that it sees. Then, the researchers were creating algorithms to detect shapes, lines, and edges in photographs.
Decades after that computer vision evolved into several subfields like signal processing, image processing, computer photometry, object recognition, and so on. Object recognition is one of the most important and challenging area in computer vision. The breakthrough in this area led to the rise of AI known today.

**Yann LeCun(1998)** was the first to develop CNN based **handwritten digits recognition(LeNet)**, and so prove the effectiveness of CNNs in image recognition.

Attribute based classification is widely used with unsupervised (classification) machine learning models. Amazon Mechanical Turks enhanced the Caltech-UCSD birds database(CUB-200-2011) by labeling the attributes, such as color, size, visual features for each photo. This makes it easy to use these attributes for birds image classification using Random Forest or similar classifiction models. Other examples are Iris flowers, Rice datasets.

**Alex Krizhevsky et al**.'s CNN variant architecture **AlexNet** for object recognition won ILSVRC-2012 (ImageNet Large Scale Visual Recognition Challenge). Highly sophisticated than LeNET, AlexNet provided significant breakthrough for object recognition.

**Google's Inception**, winner of ILSVRC-2014 predicts efficiently (*improved training time*) with high accuracy and far fewer training parameters, compared to its closest contender VGG16. 

Oxford's **Visual Geometry Group's VGG16** further improved CNNs and finished second in the ILSVRC-2014.

**Kaiming He et al.'s ResNet** (residual neural network), introduced at the ILSVRC-2015, has the residual block technique which allowed the neural network to be deeper with moderate number of parameters.

### b. Notable datasets and notebooks:
A number of image datasets are widely used in the industry. Few of them are MNIST(fashion, handwritten digits), CIFAR10, Flowers, Rice, Iris, Caltech-UCSD database(CUB-200-2011), Cats and Dogs by Microsoft. These datasets are of different types: image attributes, segmented images, grayscale images, RGB images, and so on.

**Rice classification:** https://www.kaggle.com/code/musab2077/rice-image-classification-by-tensorflow

**MNIST Fashion - clothing image classification:** https://www.tensorflow.org/tutorials/keras/classification

**Kaggle Dataset:** https://www.kaggle.com/datasets/gpiosenka/100-bird-species

**Comparison of Classification models:** https://github.com/projgb/ML_ClassificationModels

### c. Project Rationale:
With this overview, we can understand that models and datasets are somewhat related.

 *Thus, the project rationale is to design and develop a CNN-based model for RGB digital images, the main focus being the process compared to the end goal.*

# Design and Implementation : CNN-based Image classification model

**Data loading, preprocessing and splitting:**
* For this project, 3 classes, 60 images per class from CUB-200-2011 database are used.
* As image sizes vary greatly, all images are resized to CNN's standard image size of 256x256 and normalized by 1/255.
* Training and Validation sets are splited in the ratio of 90%-10%.

**CNN model design:**
CNN model is formed with Conv2D, MaxPooling2D, Flatten and Dense layers. More layers are added for further enhancements.

* **References:** 
  - Image Classification Tutorial : https://www.tensorflow.org/tutorials/images/classification
  - Image Classification Notebook : https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
  - **Image Dataset : http://www.vision.caltech.edu/visipedia/CUB-200-2011.html**


# Model evaluation Strategy:

 Models are evaluated using validation accuracy, validation loss, precision-accuracy-recall from the confusion matrix and classificaton report.

### Validation accuracy and Validation loss:
 Validation accuracy and validation loss are monitored during training.

### Confusion matrix and classification report:
 Since, this is unsupervised (classification) model, Precision, Recall, and Accuracy values from the confusion matrix and classificaton report are used to properly assess the performance of the model.
 Functions **plotaccloss** and **createcm_classrpt** are prepared to get consistent results across all the testing and all the time.

# Model Testing and Findings:

As the Base model was tested with the image dataset, siginficant impact of overfitting on the validation accuracy and validation loss is cleary seen.

Thus, Base model is improved in 2 stages: Tune1: to include 1 DropOut layer and earaly stopping.
This mitigated the overfitting half way through.

Next level of hyperparamater tuning is applied in Tune2 model.
* 0.1 validation split gives only 54 images for Training. Slightly reduce validation split to 0.05, 57 Training images.
* Add 4 image augmentations - Flip, Zoom, Translation(Shift) and Rotation, to further increase Training images.
* Increase the number of epochs to give more learning time for the model.
* Reduce image size (h x w) to 200x200 to reduce impact of any visual noise in the images.
* Add DropOuts per block also to reduce the overfitting.
* Retain the EarlyStopping.

With these enhancements, performance of the model improved drastically.

## Performance comparison of three models:
Let's compare these three model variants on the basis of evaluation metrics discussed above.

### Validation accuracy and Validation loss:



### Confusion Matrix and Classification Reports.









