package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components.{Affine, Output}
import org.smurve.yainn.data.DataIterator
import org.smurve.yainn.helpers.SGDTrainer

import scala.language.postfixOps


/**
  * Advanced strategy: Pretrain the first layer in an auto-encoder setup and use it afterwards to feed a regular network.
  * Delivers a 96.2% success rate
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  */
object AutoEncoderMNISTExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the default parameters and hyper-parameters here */
    val params1 = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 20
      override val ETA = 1e-4 // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
    }

    /** Overriding the default parameters and hyper-parameters here */
    val params2 = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 100
      override val ETA = 1e-1 // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
    }

    /** create an adapter that returns the pairs of images for the auto-encoder network*/
    val aeIterator = new DataIterator {
      private val adaptee = createIterator(params1)
      override def hasNext: Boolean = adaptee.hasNext
      override def reset(): Unit = adaptee.reset()
      override def newTestData(numTests: Int): (T, T) = adaptee.newTestData(numTests)
      override def nextMiniBatch(): (T, T) = {
        val nmb = adaptee.nextMiniBatch()
        ( nmb._1, nmb._1 )
      }
    }



    /** stack some layers to form a generative autoencoder */
    val numFeatures = 200

    val W1 = (Nd4j.rand(numFeatures, 784, params1.SEED) - 0.5) / 10.0
    val b1 = (Nd4j.rand(numFeatures, 1, params1.SEED) - 0.5) / 10.0

    val W2 = (Nd4j.rand(784, numFeatures, params1.SEED) - 0.5) / 10.0
    val b2 = (Nd4j.rand(784, 1, params1.SEED) - 0.5) / 10.0

    val nn1 = Affine("Dense", W1, b1) !! Affine("Dense", W2, b2) !! Sigmoid() !! Output(euc, euc_prime)

    /** Train the network to reproduce any image from its lower-dim intermediate encoding */
    new SGDTrainer(nn1).train(aeIterator, params1)

    val x = aeIterator.newTestData(1)._1
    val y = nn1.fp(x)

    /** the original image */
    println(visualize(x.reshape(28,28)))
    /** The image that the autoencoder produces */
    println(visualize(y.reshape(28,28)))



    /** the regular network */
    val iterator = createIterator(params2)
    val W3 = (Nd4j.rand(400, numFeatures, params1.SEED) - 0.5) / 10.0
    val b3 = (Nd4j.rand(400, 1, params1.SEED) - 0.5) / 10.0

    val W4 = (Nd4j.rand(10, 400, params1.SEED) - 0.5) / 10.0
    val b4 = (Nd4j.rand(10, 1, params1.SEED) - 0.5) / 10.0

    val nn2 = Affine("Dense", W1, b1) !! Relu() !!  Affine("Dense", W3, b3) !! Relu() !! Affine("Dense", W4, b4) !! Sigmoid() !! Output(euc, euc_prime)

    /** this network achieves 94.5% with fairly small layers 784 x 128 x 200 x 10 */
    new SGDTrainer(nn2).train(iterator, params2)

    predict(nn2, iterator.newTestData(params2.N_DEMO))

  }
}