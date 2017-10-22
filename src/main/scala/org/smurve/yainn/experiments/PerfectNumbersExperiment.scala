package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
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
      override val ETA = 1e-3  // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
    }

    /** read data from disk */
    val iterator = createIterator(params)

    /** stack some layers to form a network - check out this method! */
    val nn = createNetwork(params.SEED, 784, 1600, 200, 10)

    /** Use gradient descent to train the network */
    new SGDTrainer(List(nn)).train(iterator, params)


    val eta_x = 10
    for ( i<- 0 to 9) {

      /** just some white noise */
      val number = Nd4j.rand(params.SEED, 784).T / 100

      /** pre-define the desired classification */
      val yb = t(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
      yb(i) = 1.0

      /** fit the white noise to produce a particular classification */
      while (euc(nn.fp(number), yb) > 3e-4) { // just some reasonable threshold
        val dC_dx = nn.fbp(number, yb, number).dC_dy // fwd-bwd pass to retrieve dC_dy (which is dC_dx here...)
        number.subi(dC_dx * eta_x) // gradient descent: adjust the image a little bit
      }

      /** see that the network recognizes its own piece of art */
      predict(nn, (number, yb))
    }

  }
}