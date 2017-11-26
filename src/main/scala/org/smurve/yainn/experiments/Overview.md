
## Overview of experiments
Here, you'll find 10 scala classes that demonstrate different aspects of neural networks. You can run any of them in your IDE
or using sbt. They'll produce more or less similar output, only achieving their results by different means. The inner workings
of each experiment is defined in its own markdown document as listed below

#### [A Minimal Experiment](Ex_01_MinimalExperiment.md)
Sometimes it's really hard to debug a larger network or a new algorith when you work with non-trivial input data 
such as images. This experiment has very trivial input data made of ` 3 x 3 ` *image* shapes, yet it deals with the same neural 
network components as the other experiments do. Debugging is really big fun here.

#### [MNIST Most Simple](Ex_02_SimpleMNISTExperiment.md)
Use a single dense or *fully connected* layer representing 784 x 10 matrix and a 10-dim bias to classify the MNIST
digits 

#### [Hidden Layers](Ex_03_HiddenLayersMNISTExperiment.md)
Use one or more hidden layers to improve performance.

#### [Preprocessing as a Layer](Ex_04_PreprocessingMNISTExperiment.md)
In a component architecture, pre-processing steps may also be modelled as `Layer`s. See how we shrink the images with the
help of a non-learning layer stacked in front of the neural network

#### [Perfect Numbers](Ex_05_PerfectNumbersExperiment.md)
Back propagation is also useful to *optimize* the input - not the network - with respect to a certain cost function. 
Here, we demonstrate how we use backpropagation to optimize a random (white noise) image such that it becomes a *perfect* digit.

#### [SoftMax and ADAM](Ex_06_CompareNaiveWithADAMExperiment.md)
See how turning to advanced optimization techniques, such as the Adaptive Moment Optimizer and the Softmax activation 
together with cross-entropy cost dramatically improve convergence.

#### [Introducting Convolutional Layers](Ex_07_MinimalConvMNISTExperiment.md)
Here, we introduce a simplified convolutional layer and demonstrate how convolution in 2 dimensions can be implemented 
by multiplying the input by a large sparse matrix.

#### [More on Convolutions](Ex_08_ConvolutionalMNISTExperiment.md)
Trying to get the best accuracy with a single conv layer

#### [Initializing Weights with an AutoEncoder](Ex_09_AutoEncoderMNISTExperiment.md)
In the past, some researchers used auto-encoders to initialize the weights of a neural network to give the convergence a *heads-up*

#### [Regularization with Auto-Encoders](Ex_10_AutoEncoderForkMNISTExperiment.md)
Auto-encoders embedded in the neural network may also have a regularizing effect
 
