package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn.components.{Affine, AutoUpdatingAffine, Layer, Output}
import org.smurve.yainn.helpers.{Adam, L2RegAffineParameters, NaiveSGD, GradientDescentTrainer}
import org.smurve.yainn.{Relu, Sigmoid, x_ent, x_ent_prime}

import scala.language.postfixOps


/**
  * The inevitable, ubiquitous MNIST show case from ND4J/ND4S bricks and mortar
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  */
object Ex_6_NaiveSGDWithAdamComparisonExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 4
      override val ETA = 1e-1 // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
      val ETA_ADAM = 1e-2
    }

    /** read data from disk */
    val iterator = createIterator(params)


    /** stack some layers to form a simple naive SGD optimizer network */
    val nn = createNetwork_with_naiveSGD(params.SEED, params.ETA, 784, 800, 10)

    val nn_a = createNetwork_with_Adam(params.SEED, params.ETA_ADAM, 784, 800, 10)

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
      new GradientDescentTrainer(List(network)).train(iterator, params, verbose = false)
    }

  }

  def createNetwork_with_Adam(seed: Long, eta: Double, dims: Int*): Layer =
    (for (((left, right), index) <- dims.zip(dims.tail).zipWithIndex) yield {
      val name = s"Layer$index"
      AutoUpdatingAffine(name, new L2RegAffineParameters(left, right, 0.0, seed,
        Some(Adam(eta = eta, size_W = left * right, size_b = right)))) !!
        (if (index == dims.size - 2) Sigmoid() else Relu())
    }).reduce((acc, elem) => acc !! elem) !! Output(x_ent, x_ent_prime) //Output(euc, euc_prime)

  def createNetwork_with_naiveSGD(seed: Long, eta: Double, dims: Int*): Layer =
    (for (((left, right), index) <- dims.zip(dims.tail).zipWithIndex) yield {
      val name = s"Layer$index"
      AutoUpdatingAffine(name, new L2RegAffineParameters(left, right, 0.0, seed,
        Some(NaiveSGD(eta = eta)))) !!
        (if (index == dims.size - 2) Sigmoid() else Relu())
    }).reduce((acc, elem) => acc !! elem) !! Output(x_ent, x_ent_prime) //Output(euc, euc_prime)

}