package org.smurve.yainn.experiments

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.components.{Affine, BackPack, Layer, Output}
import org.smurve.yainn.data.MinimalDataIterator

/**
  * A minimal experiment with very small artificial data sets and few parameters.
  */
object MinimalExperiment {

  def main(args: Array[String]): Unit = {
    val seed = 1234L
    val N_TEST = 100
    val N_DEMO = 5

    //  // 3 different symbols on 3x3 images
    val NOISE = 0.4
    val BATCH_SIZE = 20
    val NUM_BATCHES = 2
    val NUM_EPOCHS = 5
    val ETA = 3e-1

    println("Creating data iterator...")
    val data = new MinimalDataIterator(NOISE, BATCH_SIZE, NUM_BATCHES, seed)
    println("Done.")

    val nn = createNetwork(9, 16, 3, seed)

    val testSet = data.newTestData(N_TEST)

    for (e <- 1 to NUM_EPOCHS) {
      while (data.hasNext) {

        val (trainingImages, trainingLabels) = data.nextMiniBatch()

        val BackPack(cost, _, grads) = nn.fbp(trainingImages, trainingLabels, trainingImages)
        val deltas = grads.map(p => {
          (p._1 * ETA, p._2 * ETA)
        })
        nn.update(deltas)

        if (!data.hasNext) println(s"Epoch Nr. $e, after $NUM_BATCHES batches: Cost=$cost")
      }
      data.reset()
      val successRate = successCount(nn, testSet).sum(1)
      println(s"Sucess rate: $successRate")
    }

    predict(nn, data.newTestData(N_DEMO))

  }

  def predict(nn: Layer, samples: (T, T)): Unit = {
    val imgs = samples._1
    val n = imgs.size(1)
    for ( i <- 0 until n ) {
      val img = imgs(->, i)
      val lbl = nn.fp(img)
      println(asString(img) + " = " + lbl + ": " + nameFor(lbl) + "\n")
    }
  }

  /**
    * create the network from randomly initialized weights and biases
    * @param seed a random seed
    * @return the first layer representing the network
    */
  def createNetwork (nx: Int, nh: Int, ny: Int, seed: Long): Layer = {

    val W0 = Nd4j.rand(nh, nx, seed) - 0.5
    val b0 = Nd4j.rand(nh, 1, seed) - 0.5
    val W1 = Nd4j.rand(ny, nh, seed) - 0.5
    val b1 = Nd4j.rand(ny, 1, seed) - 0.5

    Affine("Input", W0, b0) !! Relu() !! Affine("Hidden", W1, b1) !! Sigmoid() !! Output(euc, euc_prime)
  }

  /**
    * Count the number of true positives within a test set.
    */
  def successCount(nn: Layer, testSet: (T, T)): INDArray =
    equiv(nn.fp(testSet._1), testSet._2)


  def nameFor(classification: T): String = {
    if (equiv(classification, t(1,0,0)) == t(1)) "cross"
    else if (equiv(classification, t(0,1,0)) == t(1)) "diamond"
    else if (equiv(classification, t(0,0,1)) == t(1)) "tee"
    else "not recognized"
  }

  /** A value > .5 is a black pixel,
    *
    * @param img the image to display
    */
  def asString(img: T): String = {
    val reshaped = img.reshape(3,3)
    (for ( r <- 0 until 3 ) yield {
      (for (c <- 0 until 3) yield pixelFor (reshaped.getDouble(r, c) ) ).mkString("")
    }).mkString("\n")
  }

  def pixelFor(d: Double): String =
    if (d > .9) "88"
    else if ( d > .2) "::"
    else if ( d > .1) ". "
    else "  "
}