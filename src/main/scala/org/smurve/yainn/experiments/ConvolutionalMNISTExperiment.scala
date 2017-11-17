package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components._
import org.smurve.yainn.helpers._

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
object ConvolutionalMNISTExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the default parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 2000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass

      // 50 epochs get you up to 90%, 200 epochs up to 96.x
      override val NUM_EPOCHS = 30
      override val ETA = 3e-1 //
      val ALPHA = 0
    }

    def adam ( size_W: Int, size_b: Int, eta: Double ): Option[Updater] = {
      Some(Adam(eta = eta, size_W = size_W, size_b = size_b))
    }

    /** read data from disk */
    val iterator = createIterator(params)

    val nn =
      ShrinkAndSharpen(cut = .4) !!
        AutoUpdatingConv("Conv", new ConvParameters(5, 5, 14, 14, 40, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA)))) !!
        Relu() !!
        AutoUpdatingAffine("affine1", new L2RegAffineParameters(4000, 100, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA)))) !!
        Relu() !!
        AutoUpdatingAffine("affine2", new L2RegAffineParameters(100, 10, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA)))) !!
        Sigmoid() !!
        Output(x_ent, x_ent_prime)


    /** see that the network cannot yet do anything useful without training */
    val testSet = iterator.newTestData(params.TEST_SIZE)
    val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
    info(s"Sucess rate before training: $successRate")


    /** Use gradient descent to train the network */
    new SGDTrainer(List(nn)).train(iterator, params)


    /** Demonstrate the network's capabilities */
    predict(nn, iterator.newTestData(params.N_DEMO))
  }
}