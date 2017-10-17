package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components.{Affine, BackPack, Layer, Output}
import org.smurve.yainn.data.{MNISTDataIterator, MNISTFileLoader}

import scala.language.postfixOps

/**
  * The inevitable, ubiquitous MNIST show case from ND4J/ND4S bricks and mortar
  * Run this without any arguments to see how
  * a) data is loaded from the MNIST-provided files
  * b) a network is created consisting of affine (dense, or fully connected) layers and appropriate activation functions
  * c) the network is trained during a number of epochs
  * d) the trained network successfully classifies most of the images in the test set
  */
object MNISTExperiment extends Logging {

  def main(args: Array[String]): Unit = {

    /** The parameters and hyper-parameters */
    val seed = 12345L // always use a seed to guarantee reproducible results!
    val TEST_SIZE = 1000
    val N_DEMO = 1
    val TRAINING_SIZE = 60000 // use all 60000 images available in the MNIST dataset
    val MINI_BATCH_SIZE = 500 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
    val NUM_BATCHES = TRAINING_SIZE / MINI_BATCH_SIZE
    val NUM_EPOCHS = 2
    val ETA = 1e-1 // Learning Rate, you'll probably need to play with this, when you play with other networks.
    /** Additionally, use environment variable OMP_NUM_THREADS to control the number of threads used by BLAS */

    /** load images and labels from file */
    val loader = new MNISTFileLoader("input")
    val trainingData= (loader.readImages("train", TRAINING_SIZE), loader.readLabels("train-labels", TRAINING_SIZE))
    val testData= (loader.readImages("test", TEST_SIZE), loader.readLabels("test-labels", TEST_SIZE))
    val data = new MNISTDataIterator(trainingData, MINI_BATCH_SIZE, testData)
    val testSet = data.newTestData(TEST_SIZE)

    /** create your network */
    val nn = createNetwork(seed, 784, 2000, 200, 10)

    /** try a larger network by simply adding another layer like below.
    val nn = createNetwork(seed, 784, 1600, 200, 10) */

    /** see that the network can't do anything usefule without training */
    val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / TEST_SIZE
    info(s"Sucess rate: $successRate")

    for (e <- 1 to NUM_EPOCHS) { // for the given number of epochs
      var duration = 0L
      while (data.hasNext) { // for all mini-batches

        val (trainingImages, trainingLabels) = data.nextMiniBatch()

        /** forward-backward pass through the entire network, the entire batch in a single go, and measure time */
        val (res, time) = timed ( nn.fbp(trainingImages, trainingLabels))
        val BackPack(cost, _, grads) = res
        duration += time //

        /** multiply all gradients by a common learning rate and update the gradients with it */
        val deltas = grads.map({ case (grad_bias, grad_weight) =>
          (grad_bias * ETA, grad_weight * ETA)
        })
        nn.update(deltas)

        if (!data.hasNext) { /** Do some stats at the end of each epoch */
          println(s"Epoch Nr. $e, after $NUM_BATCHES batches: Cost=$cost")
          printstats( deltas )
        }
      }
      data.reset() /** restart from the first mini-batch */

      /** validate the networks performance with the help of the test set*/
      val successRate = successCount(nn, testSet).sumT[Double] * 100.0 / TEST_SIZE
      println(s"Net learning time: $duration ms, Sucess rate: $successRate")
    }

    /** Demonstrate the network's capabilities */
    predict(nn, data.newTestData(N_DEMO))
  }


  /**
    * create an arbitrary-length network of ReLU-activated affine layers
    * Pls forgive me and consider the recklessly functional style a recreational exercise...;-)
    * @param seed an rng seed
    * @param dims dimensions of the layers: 1st is x, last is y
    * @return a multilayer network
    */
  def createNetwork (seed: Long, dims: Int*): Layer =
    (for (((left,right), index) <- dims.zip(dims.tail).zipWithIndex) yield {
      val name = s"Layer$index"
      val W = (Nd4j.rand(right, left, seed) - 0.5) / 10.0
      val b = (Nd4j.rand(right, 1, seed) - 0.5) / 10.0
      Affine(name, W, b) !!
        (if (index == dims.size - 2) Sigmoid() else Relu())
    }).reduce((acc, elem) => acc !! elem) !! Output(euc, euc_prime)


  /**
    * Examplary: Make sure that you always have some insight into the networks inner workings
    * print summary statistics about the weight and bias updates
    * @param toProbe the
    */
  private def printstats(toProbe: List[(T,T)]): Unit = {

    def summary(s: T ): String = {
      val max = s.maxT[Double]
      val min = s.minT[Double]
      val avg = s.sumT[Double] / s.length()
      val zeros = ( s === 0 ).sumT[Double]
      s"max: $max, min: $min, avg: $avg, zeros: $zeros"
    }

    toProbe.foreach(p=>{
      println("Weights: " + summary(p._1))
      println("Biases:  " + summary(p._2))
    })
  }


  /**
    * Demonstrate the networks capabilities at the hand of some samples
    * @param nn the neural network to do the 'job'
    * @param samples the samples to be classified
    */
  def predict(nn: Layer, samples: (T, T)): Unit = {
    val imgs = samples._1
    val lbls = samples._2
    val n = imgs.size(1)
    for ( i <- 0 until n ) {
      val img = imgs(->, i)
      val pred = nn.fp(img)

      println(visualize(img.reshape(28, 28)))
      val labeledAs = (lbls(->,i).T ** t(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)).getDouble(0)

      val classidAs = toArray(pred).zipWithIndex.reduce((a, v) => if (v._1 > a._1) v else a)._2
      println(s"labeled as   : $labeledAs, classified as: $classidAs - $pred")
    }
  }


  /**
    * Count the number of true positives within a test set.
    */
  def successCount(nn: Layer, testSet: (T, T)): INDArray =
    equiv(nn.fp(testSet._1), testSet._2)


}