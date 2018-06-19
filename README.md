# CNNfruits
Deep Learning - Convolution Neural network for fruits dataset

# Data Description
1. Total Number of Images: 38409
2. Training set size: 28736 images.
3. Validation set size: 9673 images.
4. Number of classes: 60 (fruits).
5. Image size: 100x100 pixels.

# Convolution Neural Network
A CNN is comprised of one or more convolutional layers and then followed by one or more fully connected layers as in a standard multilayer neural network
1. Convolution
2. Pooling
3. Fully Connected
4. Weights

# Model
We applied convolutional neural network on the image data set with three convolutional layers
1. First layer -> convolutional layer 5 x 5 x 3 -> 32 outputs
2. Max pooling -> 2 x 2 filter with stride 2
3. Second layer -> convolution layer 5 x 5 x 128 -> 64 outputs
4. Fully connected layer with 64 inputs and 32 outputs and so on
5. This is in turn fed to the activation function, SoftMax
6. SoftMax loss layer has 32 inputs and the output is equal to the number of classes

# Keras
1. Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
2. We used TensorFlow on backend.
3. Allows for easy and fast prototyping.
4. Supports both convolutional networks and recurrent networks, as well as combinations of the two.
5. Runs seamlessly on CPU and GPU.

