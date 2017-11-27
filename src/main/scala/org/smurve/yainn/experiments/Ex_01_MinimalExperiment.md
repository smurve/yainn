## Experiment 1: Small symbols

The Scala Code: [`Ex_01_MinimalExperiment`](Ex_01_MinimalExperiment.scala)

This experiment highlights the benefits of having a known-to-be-solvable problem with very little data at hand.
Using 3 x 3 images of three classes is good enough to verify convergence and a correct implementation of gradients.
Debugging machine learning programs is notoriously cumbersome and slow, as your IDE would attempt to display the variables
at a breakpoint. These variables may contain many 100'000s or even millions of numbers. 
  Having a small problem at hand allows you to debug efficiently.


Below are the shapes, printed out at the console, "88" denotes a single pixel
```
888888
  88
  88       Tee
    
    
88  88
  88
88  88     Cross
    
    
  88
88  88
  88       Diamond
```

The experiment creates a fully connected network with a single hidden layer

```
    val W0 = Nd4j.rand(nh, nx, seed) - 0.5
    val b0 = Nd4j.rand(nh, 1, seed) - 0.5
    val W1 = Nd4j.rand(ny, nh, seed) - 0.5
    val b1 = Nd4j.rand(ny, 1, seed) - 0.5
    
    Affine("Input", W0, b0) !! Relu() !! Affine("Hidden", W1, b1) !! Sigmoid() !! Output(euc, euc_prime)
``` 

and trains it with mini-batch gradient descent:

```
    for (e <- 1 to NUM_EPOCHS) {
      while (data.hasNext) {
    
        val (trainingImages, trainingLabels) = data.nextMiniBatch()
    
        val BackPack(cost, _, grads) = nn.fbp(trainingImages, trainingLabels, trainingImages)
        val deltas = grads.map(p => {
          (p._1 * ETA, p._2 * ETA)
        })
        nn.update(deltas)
    
        if (!data.hasNext) println(s"Epoch Nr. $e, after $NUM_BATCHES batches: Cost=$cost")
      }
      data.reset()
      val successRate = successCount(nn, testSet).sum(1)
      println(s"Sucess rate: $successRate")
    }
```

You'll see, it's done in no time. I found it extremely helpful to develop all the basic foundations with the help
of this minimalistic approach.