## Experiment 2: MNIST Fully Connected with Hidden Layers

The Scala Code: [`Ex_03_HiddenLayersMNISTExperiment`](Ex_03_HiddenLayersMNISTExperiment.scala)

This experiments runs 40 epochs agains a fully connnected network with three hidden layers to achieve 96.5 percent accuracy on the test set.
It uses the utility function `createNetwork`, that you can use to create arbitrarily deep fully connected networks by simply adding more layer
dimensions. The function uses varargs on its last argument

```
    val nn = createNetwork(params.SEED, 784, 1400, 400, 200, 10)
```

On my 6 CPUs, the 40 epochs took about 8 seconds each. Time enough for a coffee. Or for checking out `createNetwork()`
an interesting use case for the map-reduce algorithm:

Pairs of adjacent numbers are mapped to layers, which are then reduced to form a network with the help of the stacking operator `!!`

```
  /**
    * create an arbitrary-length network of ReLU-activated affine layers
    * Pls forgive me and consider the ruthlessly functional style a recreational exercise...;-)
    * @param seed an rng seed
    * @param dims dimensions of the layers: 1st is x, last is y
    * @return a multilayer network
    */
  def createNetwork (seed: Long, dims: Int*): Layer =
    (for (((left,right), index) <- dims.zip(dims.tail).zipWithIndex) yield {
      val name = s"Layer$index"
      val W = (Nd4j.rand(right, left, seed) - 0.5) / 10.0
      val b = (Nd4j.rand(right, 1, seed) - 0.5) / 10.0
      Affine(name, W, b) !!
        (if (index == dims.size - 2) Sigmoid() else Relu())
    }).reduce((acc, elem) => acc !! elem) !! Output(x_ent, x_ent_prime)

```

