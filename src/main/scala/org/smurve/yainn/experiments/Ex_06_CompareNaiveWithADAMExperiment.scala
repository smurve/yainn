package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components.{AutoUpdatingAffine, Layer, Output}
import org.smurve.yainn.helpers.{Adam, GradientDescentTrainer, L2RegAffineParameters, NaiveSGD}

import scala.language.postfixOps


/**
  * The inevitable, ubiquitous MNIST show case from ND4J/ND4S bricks and mortar
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  */
object Ex_06_CompareNaiveWithADAMExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 10
      override val ETA = 1e-1 // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
      val ETA_ADAM = 1e-2
    }

    /** read data from disk */
    val iterator = createIterator(params)


    /** stack some layers to form a simple naive SGD optimizer network */
    val nn = createNetwork_with_naiveSGD(params.SEED, params.ETA, 784, 1400, 100, 10)

    val nn_a = createNetwork_with_StateOfTheArt(params.SEED, params.ETA_ADAM, 784, 1600, 100, 10)

    /** see that the network cannot yet do anything useful without training */
    for ((name, network) <- List(
      ("regular", nn),
      ("adam", nn_a)
    )) {
      info("")
      info(s"Using $name optimizer")
      val testSet = iterator.newTestData(params.TEST_SIZE)
      val successRate = successCount(network, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
      info(s"Sucess rate before training: $successRate")

      /** Use gradient descent to train the network */
      new GradientDescentTrainer(List(network)).train(iterator, params)
    }

  }

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
          Relu())
    }).reduce((acc, elem) => acc !! elem)


  def createNetwork_with_naiveSGD(seed: Long, eta: Double, dims: Int*): Layer =
    (for (((left, right), index) <- dims.zip(dims.tail).zipWithIndex) yield {
      val name = s"Layer$index"
      AutoUpdatingAffine(name, new L2RegAffineParameters(left, right, 0.0, seed,
        Some(NaiveSGD(eta = eta)))) !!
        (if (index == dims.size - 2) Sigmoid() else Relu())
    }).reduce((acc, elem) => acc !! elem) !! Output(euc, euc_prime)

}