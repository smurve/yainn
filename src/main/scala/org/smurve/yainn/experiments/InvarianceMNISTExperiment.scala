package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components._
import org.smurve.yainn.helpers.{ConvParameters, L2RegAffineParameters, SGDTrainer}

import scala.language.postfixOps


/**
  * MNIST with a convolutional layer
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of pre-processing, convolutional and affine layers
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  * e) slow things become when convolutional layers are implemented the "naive" way.
  */
object InvarianceMNISTExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the default parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 2000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 20
      override val ETA = 3e-3
      val ALPHA = 5e-2
    }

    /** read data from disk */
    val iterator = createIterator(params)

    val nn =
      ShrinkAndSharpen(cut = .4) !!
        AutoUpdatingConv("Conv", ConvParameters(10, 8, 14, 14, 20, params.ETA, params.ALPHA, params.SEED)) !!
        Relu() !!
        AutoUpdatingAffine("affine3", new L2RegAffineParameters(700, 300, params.ETA, params.ALPHA, params.SEED)) !!
        Relu() !!
        AutoUpdatingAffine("affine3", new L2RegAffineParameters(300, 100, params.ETA, params.ALPHA, params.SEED)) !!
        Relu() !!
        AutoUpdatingAffine("affine3", new L2RegAffineParameters(100, 10, params.ETA, 0.0, params.SEED)) !!
        Sigmoid() !! Output(euc, euc_prime)



    /** see that the network cannot yet do anything useful without training */
    val testSet = iterator.newTestData(params.TEST_SIZE)
    val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
    info(s"Sucess rate before training: $successRate")

    /** Use gradient descent to train the network */
    new SGDTrainer(List(nn)).train(iterator, params)

    /** Demonstrate the network's capabilities */
    predict(nn, iterator.newTestData(params.N_DEMO))

    displayPerfectDigits(nn, 1e-0, 100, params.SEED)
  }


}