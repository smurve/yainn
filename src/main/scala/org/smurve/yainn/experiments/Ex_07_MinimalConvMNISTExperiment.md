## Experiment 7: A minimal convolutional network

The Scala Code: [`Ex_07_MinimalConvMNISTExperiment`](Ex_07_MinimalConvMNISTExperiment.scala)

Our implementation of convolutional networks will be very, very basic. For a general understanding of convolutions
I suggest your read [Ujjwal Karn's Blog](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) on convnets.
For even more in-depth understanding of the underlying statistical interpretations, Chris Olah has a collection of
[insightful visualizations](http://colah.github.io/).


In this experiment we demonstrate the with a convolution of 20 features of dimensions 5x5 each, a single fully connected
layer can achieve 96.4% accuracy after 10 epochs. That means essentially, that the dense layer can make much more sense
of 20 extracted features than of the pixels themselves.

Alas, due to the limitations of our implementation (it's only 2d), we cannot use more than a single convolutional layer. Anyway, this
project is meant to be recreational, or educational at best... ;-)
