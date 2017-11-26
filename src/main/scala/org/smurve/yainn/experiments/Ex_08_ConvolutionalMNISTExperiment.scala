package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components._
import org.smurve.yainn.helpers._
import org.nd4j.linalg.ops.transforms.Transforms._

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
object Ex_08_ConvolutionalMNISTExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the default parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass

      // 50 epochs get you up to 90%, 200 epochs up to 96.x
      override val NUM_EPOCHS = 10
      override val ETA = 1e-2 //
      val ALPHA = 0
    }

    def adam ( size_W: Int, size_b: Int, eta: Double ): Option[Updater] = {
      Some(Adam(eta = eta, size_W = size_W, size_b = size_b))
    }

    /** read data from disk */
    val iterator = createIterator(params)

    val (fh, fw, fn) = (5, 5, 20) // fn feature matrices of size fh x fw
    val (fho, fwo) = (14 - fh + 1, 14 - fw + 1) // produce output of size fho x fwo from a 14 x 14 input

    // Note that we use softmax with cross-entropy here for the first time. Another massive performance improvement
    val nn =
      ShrinkAndSharpen(cut = .4) !!
        AutoUpdatingConv("Conv", new ConvParameters(fw, fh, 14, 14, fn, params.ALPHA, params.SEED, adam(fh * fw * fn, fn, params.ETA))) !!
        Relu() !!
        AutoUpdatingAffine("affine1", new L2RegAffineParameters(fn * fho * fwo, 100, params.ALPHA, params.SEED, adam(fn * fho * fwo * 100, 100, params.ETA))) !!
        Relu() !!
        AutoUpdatingAffine("affine2", new L2RegAffineParameters(100, 10, params.ALPHA, params.SEED, adam(100 * 10, 10, params.ETA))) !!
        SoftMaxCrossEntropy()

    /** see that the network cannot yet do anything useful without training */
    val testSet = iterator.newTestData(params.TEST_SIZE)
    val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
    info(s"Sucess rate before training: $successRate")


    /** Use gradient descent to train the network */
    new GradientDescentTrainer(List(nn)).train(iterator, params)


    /** Demonstrate the network's capabilities */
    val testData = iterator.newTestData(params.N_DEMO)
    predict(softmax(nn.fp(testData._1).T).T, testData)
  }
}