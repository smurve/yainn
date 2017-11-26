package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4s.Implicits._
import org.smurve.yainn.components.{AutoEncoderFork, AutoUpdatingAffine, Output}
import org.smurve.yainn.helpers.{L2RegAffineParameters, NaiveSGD, GradientDescentTrainer}
import org.smurve.yainn._

import scala.language.postfixOps


/**
  * Uses a fork to combine a simple affine autoencoder with a regular feed forward network, note that the cost function
  * of this network balances prediction and autoencoding costs!
  *
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  */
object Ex_10_AutoEncoderForkMNISTExperiment extends AbstractMNISTExperiment with Logging {

  def main(args: Array[String]): Unit = {

    /** Overriding the parameters and hyper-parameters here */
    val params = new Params() {
      override val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
      override val NUM_EPOCHS = 50
      override val ETA = 1e-1 // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
      val ETA_AE = 1e-2 // learning rate for layers involved in auto-encoding
      val ALPHA = 1e-3 // L2 regularization factor
    }

    /** read data from disk */
    val iterator = createIterator(params)

    val ae_tail =
      AutoUpdatingAffine("AE_Tail", new L2RegAffineParameters(200, 784, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA_AE)))) ::
      Output(euc, euc_prime)


    val input = AutoUpdatingAffine("Input", new L2RegAffineParameters(784, 200, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA_AE))))
    val tail =
      Relu() ::
      AutoUpdatingAffine("Hidden1", new L2RegAffineParameters(200, 100, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA)))) ::
      Relu() ::
      AutoUpdatingAffine("Hidden2", new L2RegAffineParameters(100, 10, params.ALPHA, params.SEED, Some(NaiveSGD(params.ETA)))) ::
      Sigmoid() ::
      Output(x_ent, x_ent_prime)


    /** stack some layers to form a network */
    val nn =
      input ::
      AutoEncoderFork(ae_tail, .5) ::
      tail

    /** see that the network cannot yet do anything useful without training */
    val testSet = iterator.newTestData(params.TEST_SIZE)
    val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
    info(s"Sucess rate before training: $successRate")


    /** Use gradient descent to train the network */
    new GradientDescentTrainer(List(nn)).train(iterator, params)


    /** Demonstrate the network's capabilities */
    val testData = iterator.newTestData(params.N_DEMO)
    predict(nn.fp(testData._1), testData)

    /** can't use the auto-encoder here, don't need to, anyway */
    displayPerfectDigits(input :: tail, 1e-0, 100, params.SEED)

  }
}