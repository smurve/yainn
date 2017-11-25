package org.smurve.yainn.experiments

import grizzled.slf4j.Logging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components.{Affine, Layer, Output}
import org.smurve.yainn.data.{MNISTDataIterator, MNISTFileLoader}
import org.nd4j.linalg.ops.transforms.Transforms._

import scala.language.postfixOps

/**
  * Default params to avoid unnecessary duplication
  */
class Params () {
  val SEED = 12345L // always use a seed to guarantee reproducible results!
  val TEST_SIZE = 1000 // could be 10'000 at max
  val N_DEMO = 10 // typically enough just to demo
  val TRAINING_SIZE = 60000 // use all 60000 images available in the MNIST dataset
  val MINI_BATCH_SIZE = 1000 // parallelize: use mini-batches of 1000 in each fwd-bwd pass
  val NUM_EPOCHS = 2 // Take your time and allow the network to converge.
  val ETA = 1e-4  // Learning Rate, you'll probably need adapt, when you experiment with other network designs.
  def NUM_BATCHES: Int = TRAINING_SIZE / MINI_BATCH_SIZE
}


/**
  * Abstract base class for MNIST experiments
  */
class AbstractMNISTExperiment extends Logging {

  /**
    * load images and labels from file
    * @param params a parameter set
    * @return a data iterator with the entire training and test set in memory
    */
  def createIterator(params: Params): MNISTDataIterator = {
    val loader = new MNISTFileLoader("input")
    val trainingData = (loader.readImages("train", params.TRAINING_SIZE),
      loader.readLabels("train-labels", params.TRAINING_SIZE))
    val testData = (loader.readImages("test", params.TEST_SIZE), loader.readLabels("test-labels", params.TEST_SIZE))
    new MNISTDataIterator(trainingData, params.MINI_BATCH_SIZE, testData)
  }


  /**
    * create an arbitrary-length network of ReLU-activated affine layers
    * Pls forgive me and consider the ruthlessly functional style a recreational exercise...;-)
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
    }).reduce((acc, elem) => acc !! elem) !! Output(x_ent, x_ent_prime) //Output(euc, euc_prime)


  /**
    * Examplary: Make sure that you always have some insight into the networks inner workings
    * print summary statistics about the weight and bias updates
    * @param toProbe the
    */
  def printstats(toProbe: List[(T,T)]): Unit = {

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
    * @param samples the samples to be classified and compared to their true labels
    */
  def predict(nn: Layer, samples: (T, T)): Unit = {
    val imgs = samples._1
    val lbls = samples._2
    val n = imgs.size(1)
    for ( i <- 0 until n ) {
      val img = imgs(->, i)
      val pred = nn.fp(img)

      val tinyImage = sharpen(pool2by2(28) ** img * 0.25, 0.3).reshape(14, 14)

      println(visualize(tinyImage))

      /** find the single '1' in the label vector */
      val labeledAs = (lbls(->,i).T ** t(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)).getDouble(0)

      /** we interpret the output vector's largest component as the classification of our network */
      val classidAs = toArray(pred).zipWithIndex.reduce((a, v) => if (v._1 > a._1) v else a)._2

      println(s"labeled as   : $labeledAs, classified as: $classidAs - $pred")
    }
  }


  /**
    * Count the number of true positives within a test set.
    */
  def successCount(nn: Layer, testSet: (T, T)): INDArray =
    equiv(nn.fp(testSet._1), testSet._2)


  /**
    * derive 'perfect' digits by optimizing white noise, until it is recognized as one of the images
    */
  def displayPerfectDigits(nn: Layer, eta: Double, nmax: Int, seed: Long): Unit = {
    for ( i<- 0 to 9) {

      /** just some white noise */
      var digit = Nd4j.rand(seed, 784).T / 100

      /** pre-define the desired classification */
      val yb = t(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
      yb(i) = 1.0

      /** fit the white noise to produce a particular classification */
      for  {
        _ <- 1 to nmax if euc(nn.fp(digit), yb) > 1e-2
      } {
        //val c = euc(nn.fp(digit), yb)
        //println(s"Cost: $c")
        val dC_dx = nn.fbp(digit, yb, digit, update = false).dC_dy // fwd-bwd pass to retrieve dC_dy (which is dC_dx here...)
        digit = relu(digit - dC_dx * eta)// gradient descent: adjust the image a little bit
      }
      val scale = 1.0 / (digit.maxNumber().doubleValue()+1e-6)

      /** see that the network recognizes its own piece of art */
      predict(nn, (digit * scale, yb))
    }

  }
}