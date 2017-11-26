
## YAINN: Design and Architecture

### High Performance Linear Algebra
We're using [ND4J](http://nd4j.org/), more precisely its Scala wrapper [ND4S](http://nd4j.org/scala), as it comes with
GPU support and is - to my best knowledge - the currently (Nov 2017) most advanced Scala library for high performance linear algebra.
The Scala wrapper overrides operators to provide a natural mathematical algebra feeling to the programmer. Note that this is
definitely one of the rare cases where operator overloading is almost undisputable... ;-)

Throughout the code we use the type alias `T`, defined as 
```
type T = org.nd4j.linalg.api.ndarray.INDArray
```
to refer to ND4J's `INDArray` interface that hides the platform-specific implementations driving our linear algebra. ND4J
automatically determines what implementation is the fastest you can get on your machine. 

### The Layer concept
In the entire code base we're using the trait [Layer](../components/Layer.scala) that - amongst others implements the following methods:
```
def fp ( x: T): T
def fbp(x: T, yb: T, orig_x: T, update: Boolean = true): BackPack 
def !!(rhs: Layer): Layer
```
`fp(x)` is a forward-pass that takes a m x n input tensor, representing a mini-batch of m n-dimensional column vectors, and returns
the prediction, based on the current set of parameters

`fbp(x, yb, orig_x)` implements the forward-backward pass in a single go.

`def !!(rhs: Layer): Layer` creates a stack of two layers, which is itself a `Layer`, too

A neural network is any stack/chain of layers that ends with an output layer, the latter providing the cost function. So, a 
neural network in our code base is simply represented by the input layer. There's no such class as a `NeuralNetwork`, or so.

Here's the code for the smallest meaningful network. `Affine` is a more mathematical term for dense layer, and `Euclidean()` 
is an output layer providing a cost function based on the euclidean distance between the predictions and the resp. labels.

```
    val W = (Nd4j.rand(10, 784, params.SEED) - 0.5) / 10.0
    val b = (Nd4j.rand(10, 1, params.SEED) - 0.5) / 10.0

    val network = Affine("Dense", W, b) !! Sigmoid() !! Euclidean()
```
This very simple network achieves a *ridiculously* low accuracy of 84.6% after 40 episodes. We'll improve on that, I promise.



## Training with the `GradientDescentTrainer`
Here's the slightly simplified code for the trainer 
```
    for (e <- 1 to NUM_EPOCHS) {
      while (iterator.hasNext) { // for all mini-batches
        val (trainingImages, trainingLabels) = iterator.nextMiniBatch()

        val BackPack(cost, _, grads) = nn.fbp(trainingImages, trainingLabels, ...))
    
        val deltas = grads.map({ case (grad_bias, grad_weight) =>
            (grad_bias * ETA, grad_weight * ETA)
        nn.update(deltas)
      }
      iterator.reset()
      val successRate = successCount(nn, ...)
      ...
    }
```
