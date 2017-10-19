package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components.{Affine, Output}
import org.smurve.yainn.helpers.SGDTrainer

import scala.language.postfixOps


/**
  * The inevitable, ubiquitous MNIST show case from bricks and mortar with a single dense layer
  * With the help of ND4J/ND4S linear algebra abstractions.
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  */
object SimpleMNISTExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 100
      override val ETA = 1e-3 // Learning Rate, you'll probably need to play with this, when you play with other networks designs.
    }

    /** read data from disk */
    val iterator = createIterator(params)


    /** stack some layers to form a network*/
    val W = (Nd4j.rand(10, 784, params.SEED) - 0.5) / 10.0
    val b = (Nd4j.rand(10, 1, params.SEED) - 0.5) / 10.0

    val nn = Affine("Dense", W, b) !! Sigmoid() !! Output(euc, euc_prime)


    /** see that the network cannot yet do anything useful without training */
    val testSet = iterator.newTestData(params.TEST_SIZE)
    val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
    info(s"Sucess rate before training: $successRate")


    /** Use gradient descent to train the network */
    new SGDTrainer(nn).train(iterator, params)


    /** Demonstrate the network's capabilities */
    predict(nn, iterator.newTestData(params.N_DEMO))
  }
}