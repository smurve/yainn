## Experiment 1: Single layer MNIST

The Scala Code: [`Ex_02_SimpleMNISTExperiment`](Ex_02_SimpleMNISTExperiment.scala)

This experiment creates the smallest possible neural network, that just maps each of the 784 input pixels straight 
to the classifications. So, each weight of the first row of the 10 x 784 matrix basically answers the question: If pixel (x,y) is set, 
what is the chance of the image of being digit zero, and so on for the rest of the rows.

```
    val W = (Nd4j.rand(10, 784, params.SEED) - 0.5) / 10.0
    val b = (Nd4j.rand(10, 1, params.SEED) - 0.5) / 10.0
    
    val nn = Affine("Dense", W, b) !! Sigmoid() !! Euclidean()
```

Here, we introduce the `GradientDescentTrainer` for the first time. This trainer will always update the weights and biases
by subtracting the gradients times the learning rate. Later on we use *auto-updating* layers to introduce advanced techniques such
as Kingma and Ba's extremely successful [Adaptive Moments](https://arxiv.org/pdf/1412.6980.pdf) algorithm.

Using 6 CPUs of my Mac Pro, the 40 epochs take some 30 seconds before the network stops improving.
After training the network, we take the some of the digits and display them with their true labels and predictions:

```
 - --------------------------
|                            |
|            ''''            |
|        ;;@@00@@;;          |
|        @@    ;;cc          |
|              @@::          |
|            cc00            |
|          ''@@              |
|          @@;;              |
|        ;;@@                |
|        @@;;                |
|        @@@@@@@@OO00@@00cc  |
|              ::''          |
|                            |
|                            |
 ----------------------------
 labeled as   : 2.0, classified as: 2 - [0.13,  0.04,  0.61,  0.27,  0.00,  0.16,  0.34,  0.00,  0.09,  0.01]
```

You can see the activations for each digit. The above classification is for class "2", because the activation at position 2 is the largest of all 10.