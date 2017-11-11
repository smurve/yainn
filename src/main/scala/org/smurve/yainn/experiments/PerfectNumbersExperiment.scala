package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.smurve.yainn.helpers.SGDTrainer

import scala.language.postfixOps


/**
  * Using gradients differently: Let the trained network draw an image for a given digit
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained is used to optimize a white-noise image such that it produces a given classification
  */
object PerfectNumbersExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 5
      override val ETA = 1e-1 // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
    }

    /** read data from disk */
    val iterator = createIterator(params)

    /** stack some layers to form a network - check out this method! */
    val nn = createNetwork(params.SEED, 784, 1600, 200, 10)

    /** Use gradient descent to train the network */
    new SGDTrainer(List(nn)).train(iterator, params)


    displayPerfectDigits(nn, 1e-0, 100, params.SEED)
  }
}