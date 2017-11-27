## Experiment 6: Improving convergence

The Scala Code: [`Ex_06_CompareNaiveWithADAMExperiment`](Ex_06_CompareNaiveWithADAMExperiment.scala)

In this experiment, at last we're going to let some of of the more naive technologies behind. In this example, we
create two networks, one with euclidean and sigmoid and naive minibatch gradient descent, and another one with a better
final activation and cost function, with a state-of-the-art updating algorithm and learning rate decay. Observe how the 
advanced network dramatically outperforms our former naive approach with a whooping 97.6% compared to 79.4% after 10 epochs.

```
  def createNetwork_with_StateOfTheArt(seed: Long, eta: Double, dims: Int*): Layer =
    (for (((left, right), index) <- dims.zip(dims.tail).zipWithIndex) yield {
      val name = s"Layer$index"
  
      // This affine layer performes the update automatically with every call to fbp(...)
      AutoUpdatingAffine(name,
  
        // L2 regularation with a weight coefficient alpha of 0.001
        new L2RegAffineParameters(left, right, 0.001, seed,
  
        // Adam optimizer with decaying learning rate
        Some(Adam(eta = eta,
          decay=0.999,
          size_W = left * right, size_b = right)))) !!
        (if (index == dims.size - 2)
  
          // softmax activation is included in the cross-entropy output layer
          SoftMaxCrossEntropy()
        else
          Relu()) // between every two layers
    }).reduce((acc, elem) => acc !! elem)
```

### Final Activation and cost function
You may have wondered if the choice of sigmoid and the euclidean cost function is particularly good. As a matter of fact, that choice is not
optimal. State of the art networks use the so-called softmax activation with the cross-entropy cost. The massive improvement
comes from the fact that the initial gradients from the output layers are much more effectively driving gradient descent into the 
desired direction. Michael Nielson describes this effect comprehensively in [Chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html)
of his web book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
The most concise mathematical discussion of that topic that I have found is on 
[Peter Roelants' Blog](http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/). 

### ADAM: Updating with Adaptive Moments
There are quite a number of algorithms around to improve the parameter updates. Sebastian Ruder has compiled a wonderful overview
in his blog [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html).
Amongst all the choices, the probably most popular algorithm is Kingma and Ba's [ADAM](https://arxiv.org/pdf/1412.6980.pdf), 
that's why I chose to implement it here, also. 

### Learning rate decay
When the learning rate is to high, the descent path may bounce around the minimum and may even diverge totally. Sometimes this
bouncing (not the divergence, though) can be observed as the descent path gets closer to the minimum. To avoid this bouncing,
it's a common practice to decrease the learning, hence slowing down at arrival. 

### L2 Regularization
Sometimes, networks tend to fit perfectly to the training set but increasingly fail on the held-out test set. This is called overfitting.
Various methods have been tried to come by this problem, one of which is weight decay or L2 regularization, also described in aforementioned
[Chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html). 



