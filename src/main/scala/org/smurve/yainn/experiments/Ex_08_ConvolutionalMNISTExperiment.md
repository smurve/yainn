## Experiment 8: More on Convolutional network

The Scala Code: [`Ex_08_ConvolutionalMNISTExperiment`](Ex_08_ConvolutionalMNISTExperiment.scala)

This experiment builds on a preprocessor, followed by a convolutional layer followed by two fully connected layers. You can play with the main parameters: The dimensions of the 
convolutional features, the number of features and the number of neurons in the hidden layer. All dependent parameters will be calculated for you.

```
    val (fh, fw, fn, nh) = (7, 7, 30, 400) // fn feature matrices of size fh x fw, and nh in the hidden layer

    // Note that we use softmax with cross-entropy here for the first time. Another massive performance improvement
    val (fho, fwo) = (14 - fh + 1, 14 - fw + 1) // produce output of size fho x fwo from a 14 x 14 input
    val nn =
      ShrinkAndSharpen(cut = .4) !!
        AutoUpdatingConv("Conv", new ConvParameters(fw, fh, 14, 14, fn, params.ALPHA, params.SEED, adam(fh * fw * fn, fn, params.ETA))) !!
        Relu() !!
        AutoUpdatingAffine("affine1", new L2RegAffineParameters(fn * fho * fwo, nh, params.ALPHA, params.SEED, adam(fn * fho * fwo * nh, nh, params.ETA))) !!
        Relu() !!
        AutoUpdatingAffine("affine2", new L2RegAffineParameters(nh, 10, params.ALPHA, params.SEED, adam(nh * 10, 10, params.ETA))) !!
        SoftMaxCrossEntropy()

```

The above settings for the hyper-parameters achieve an accuracy of 97.5% after 30 batches. Each batch takes about 50 seconds using 6 cores of my
Mac Pro. So make sure you bring some patience along.