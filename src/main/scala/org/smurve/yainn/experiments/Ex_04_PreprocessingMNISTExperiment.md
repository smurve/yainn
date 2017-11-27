## Experiment 4: Using a preprocessing layer

The Scala Code: [`Ex_03_HiddenLayersMNISTExperiment`](Ex_03_HiddenLayersMNISTExperiment.scala)

```
    val nn = ShrinkAndSharpen(cut = .4) !! createNetwork(params.SEED, 196, 400, 100, 10)

```

Here, we use a special preprocessing layer, that shrinks the image down to 14 x 14 pixels and then
sets all pixels below a certain threshold to zero, thereby sharpening the image. Admittedly, the usage of ReLU for
sharpening is disputable. This was just an easy choice, that does the job of demonstrating the use of additional
non-learning layers for pre-processing.  