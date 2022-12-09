# vignette-cnn-facial-recognition

Preparing and Augmenting Image Data for CNNs; created as a class project for PSTAT197A in Fall 2022.

Contributors: Allester Ramaryat, Arthur Starodynov, Kaylin Roberts

Abstract: breif description in a few sentences, of your vingette topic, example data, and outcomes.

For this project we created two CNN's for image classification. The first CNN a binary model used to identify sex and the second CNN is a multi-class model used to label images from 10 classes. For the sex binary model we used data we used the IMBD-WIKI data set which had 500,000+ face images with sex and age labels. We used the CIFAR-10 data for our multi-class model which consists of 60,000 32x32 colored images with 10 classes. We used the openCV package to blur, sharpen, use edge detection, and combinations of those filters to make it easier for the model to learn from the data. After comparing multiple models with the different augmented data sets we concluded that **SHARPENING** the images gave the best model for multi-class with **83% ACCURACY ON THE TEST DATA**. Our sex recognition model achieved about 82% accuracy on the test data. 

Repository Contents: An explanation of the directory structure of the repository

Referenece List: 2 or more references to learn more about your topic
1. https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/
2. https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37

Sex Data Set: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Gender Classifier: https://www.kaggle.com/code/vijayshankar756/genderclassifier

Multi-Class Data: https://www.cs.toronto.edu/~kriz/cifar.html
