# Automatic-Image-Captioning

In this project, I have created a neural network architecture to automatically generate captions from images. The network is a CNN-RNN model consisting of a pre trained ResNet50 model architecture that acts as an Encoder that generates a feature vector of a preprocessed input image which is then fed as input to a connecting decoder RNN that generates captions for that image.

                                      The CNN-RNN Model for Image Caption generation

<img src="images/encoder-decoder.png"> <br>

I have structured this project as a series of 3 notebooks in a sequential order:
* Understanding Dataset and Preprocessing
* Training the CNN + RNN Model
* Inference - Testing the model

## The Dataset
The model is trained on The Microsoft Common Objects in Context (MS COCO) dataset is a large-scale dataset for scene understanding. The COCO dataset is one of the largest, publicly available image datasets The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. You can read more about the dataset on the [website](http://cocodataset.org/#home).

## The Notebooks
You can refer to each of the notebooks one by one for the complete process outline, that has been documented step by step in each notebook. The first notebook describes the data loading and preprocessing steps which uses helper functions stored in data_loader.py file and vocabulary.py file.
The model.py file contains the code for model architecture.

The LSTM decoder: In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!

<img src="images/image-captioning.png">

