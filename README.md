# vignette-cnn-facial-recognition

*This is an introduction to image augmentation, feature extraction, and convolutional neural networks for  binary and multi classification; created as a class project for PSTAT197A in Fall 2022.*

**Contributors:** Allester Ramayrat, Arthur Starodynov, Kaylin Roberts

### Abstract

To begin, we go over image augmentation through explaining how certain kernals affect the output of convolutional layers. We used the openCV package to blur, sharpen, use edge detection, and combinations of those filters to demonstrate this concept.

Then, we created two deep convolutional neural network architectures for binary and multi image classification. 

Our data for binary classification comes from the [Male and Female Faces Dataset](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset). This dataset contains "2.7k pictures of Male and Female faces respectively, covering multiples ethnicities and age groups (12-13% data belongs old people in both the datasets)."

Our binary classificatiion model takes inspiration from the AlexNet CNN -- an 8-layer deep CNN with 5 convolutional layers and 3 dense layers. We reduced the number of kernels and the size of the dense layer to for account for computing limitations and output requirements. This model predicts the sex of test images with an test accuracy of 82.5%.
 
For our multi-class model we used the CIFAR-10 dataset that comes from the keras.datasets module which consists of 60,000 32x32 colored images with 10 classes. We created a similiar CNN that classified on 10 classes. 

Afterwards, we performed image augmentation on the training and test images to see if it improves accuracy. After comparing multiple models with the different augmented data sets we concluded that there was slight  change when augmenting the training and test images. The model that was trained on the original data achieved an 80.2% accuracy on the test data while the model that was trained on the augmented data achieved an 81.3% accuracy on the augmented test data. 

### Repository Contents
`main`
> `data` contains the binary face data used for binary sex classification
>> `faces` 
>>> `female`
>>> `male`

> `scripts` contains .py and .ipynb files with code used to create the vignette
>> `binary classification cnn`
>>> image preprocessing, binary classification face data, feature extraction

>> `image augmentation`
>>> OpenCV, Guassian Blur, Edge Detection, Image Sharpening

>> `multi classification`
>>> multi classification on CIFAR10 data, comparison of model with augmented (sharpened) training data

> `README.md`
> `LICENSE.md`
> `vignette.html`
> `vignette.ipynb` 

# Datasets

1. Gender Classifier: https://www.kaggle.com/code/vijayshankar756/genderclassifier
2. CIFAR10 Multi Classification: https://www.cs.toronto.edu/~kriz/cifar.html

# References
1. https://www.learndatasci.com/tutorials/convolutional-neural-networks-image-classification/
2. https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37
3. https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
4. https://www.tensorflow.org/guide/keras/sequential_model
5. https://keras.io/guides/functional_api/
