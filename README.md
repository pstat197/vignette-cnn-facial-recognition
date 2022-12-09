# vignette-cnn-facial-recognition

Preparing and Augmenting Image Data for CNNs; created as a class project for PSTAT197A in Fall 2022.

Contributors: Allester Ramayrat, Arthur Starodynov, Kaylin Roberts

Abstract: breif description in a few sentences, of your vingette topic, example data, and outcomes.

For this project we created two CNN's for image classification. The first CNN a binary model used to identify sex and the second CNN is a multi-class model used to label images from 10 classes. For the sex binary model we used data we used the IMBD-WIKI data set which had 500,000+ face images with sex and age labels. We used the CIFAR-10 data for our multi-class model which consists of 60,000 32x32 colored images with 10 classes. We used the openCV package to blur, sharpen, use edge detection, and combinations of those filters to make it easier for the model to learn from the data. After comparing multiple models with the different augmented data sets we concluded that there was no significant change when augmenting the images thus we used the original data and got about 82% accuracy on the test data. Our sex recognition model achieved about 82% accuracy on the test data. 

Folders - 
Data: Consists of both data sets used for this vignette
Scripts: contains .py files for all the code done in this vignette
Multi-Class CNN Vignette: 
    - Multi-Class CNN Vignette - the FINAL Vignette, contains all the steps and explanations of what we did
    - Image Modification: Implemntation of Gaussian Blur, Edge Detectiong, and Image Sharpening of Image Data
    - Multi-Class CNN with Sharpened Images: Used the same model as our final vignette but used sharpened images from our data      set. 

# References
1. https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/
2. https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37
3. https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
4. https://www.tensorflow.org/guide/keras/sequential_model
5. https://keras.io/guides/functional_api/



Gender Classifier: https://www.kaggle.com/code/vijayshankar756/genderclassifier

Multi-Class Data: https://www.cs.toronto.edu/~kriz/cifar.html
